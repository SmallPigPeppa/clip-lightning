import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from .encoders import ImageEncoder, ProjectionHead, TextEncoder
import copy


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            image_encoder_alias: str,
            text_encoder_alias: str,
            image_encoder_pretrained: bool = True,
            image_encoder_trainable: bool = True,
            text_encoder_trainable: bool = True,
            image_embedding_dims: int = 2048,
            text_embedding_dims: int = 768,
            projection_dims: int = 256,
            dropout: float = 0.0,
            temperature: float = 1.0,
            weight_decay: float = 0.0,
            head_lr: float = 1e-3,
            image_encoder_lr: float = 1e-4,
            text_encoder_lr: float = 1e-5,
            lr_warmup_epochs: int = 5,
            batch_size: int = 32,
            old_checkpoint_path: str = None,
            current_task: int = 0,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_alias,
            pretrained=image_encoder_pretrained,
            trainable=image_encoder_trainable,
        )
        self.text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.save_hyperparameters()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.tensor([temperature]) * np.log(1 / 0.07))
        self.val_img_feats = []
        self.val_text_feats = []
        self.save_hyperparameters()

    def _compute_losses(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor,
            # temperature: float = 0.1,
            # extra_pos_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes SimCLR's loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.

        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
            temperature (float): temperature factor for the loss. Defaults to 0.1.
            extra_pos_mask (Optional[torch.Tensor]): boolean mask containing extra positives other
                than normal across-view positives. Defaults to None.

        Returns:
            torch.Tensor: SimCLR loss.
        """

        device = self.device

        b = z1.size(0)
        z = torch.cat((z1, z2), dim=0)
        z = F.normalize(z, dim=-1)

        logits = torch.einsum("if, jf -> ij", z, z) * self.logit_scale.exp()
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
        pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        pos_mask[:, b:].fill_diagonal_(True)
        pos_mask[b:, :].fill_diagonal_(True)

        # # if we have extra "positives"
        # if extra_pos_mask is not None:
        #     pos_mask = torch.bitwise_or(pos_mask, extra_pos_mask)

        # all matches excluding the main diagonal
        logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        # loss
        loss = -mean_log_prob_pos.mean()
        return loss

    def forward(self, inputs):
        image_features = self.image_encoder(inputs["image"])
        text_features = self.text_encoder(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings

    def configure_optimizers(self):
        parameters = [
            {
                "params": self.image_encoder.parameters(),
                "lr": self.hparams.image_encoder_lr,
                "weight_decay": self.hparams.weight_decay},
            {
                "params": [self.logit_scale],
                "lr": self.hparams.head_lr,
                "weight_decay": self.hparams.weight_decay},

            {
                "params": self.text_encoder.parameters(),
                "lr": self.hparams.text_encoder_lr,
                "weight_decay": self.hparams.weight_decay},
            {
                "params": itertools.chain(
                    self.image_projection.parameters(),
                    self.text_projection.parameters(),
                ),
                "lr": self.hparams.head_lr,
                "weight_decay": self.hparams.weight_decay,
            },
        ]
        optimizer = optim.AdamW(parameters, weight_decay=self.hparams.weight_decay)
        base_lr = min(self.hparams.image_encoder_lr, self.hparams.text_encoder_lr, self.hparams.head_lr)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.lr_warmup_epochs,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=0.01 * base_lr,
            eta_min=0.01 * base_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings)
        train_loss = self.all_gather(loss)
        self.log("train/clip_loss", train_loss.mean())
        return loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings)
        val_loss = self.all_gather(loss)
        self.log("val/clip_loss", val_loss.mean())

        # for clip metrics
        self.val_img_feats.append(image_embeddings)
        self.val_text_feats.append(text_embeddings)
        return loss

    def on_validation_epoch_end(self):
        all_image_features = torch.cat(self.val_img_feats)
        all_text_features = torch.cat(self.val_text_feats)
        val_metrics = self.get_clip_metrics_cpu(
            image_features=all_image_features,
            text_features=all_text_features,
            logit_scale=self.logit_scale.exp(),
        )
        self.log_dict(val_metrics)
        self.val_img_feats.clear()
        self.val_text_feats.clear()

    def get_clip_metrics(self, image_features, text_features, logit_scale=1.0):
        metrics = {}
        logits_per_image = (logit_scale * image_features @ text_features.t())
        logits_per_text = logits_per_image.t()
        logits = {"val/image_to_text": logits_per_image, "val/text_to_image": logits_per_text}
        ground_truth = torch.arange(len(text_features)).view(-1, 1).to(self.device)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True).to(self.device)
            preds = torch.where(ranking == ground_truth)[1]
            for k in [1]:
                metrics[f"{name}_R@{k}"] = (preds < k).float().mean() * 100  # Convert recall to percentage

        return metrics

    def get_clip_metrics_cpu(self, image_features, text_features, logit_scale=1.0):
        metrics = {}
        logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
        logits_per_text = logits_per_image.t().detach().cpu()

        logits = {"val/image_to_text": logits_per_image, "val/text_to_image": logits_per_text}
        ground_truth = torch.arange(len(text_features)).view(-1, 1)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == ground_truth)[1]
            preds = preds.detach().cpu().numpy()
            for k in [1]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k) * 100  # Convert recall to percentage

        return metrics
