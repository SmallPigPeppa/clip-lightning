import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from clip_openai.model import my_load
import copy


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'RN50',
            download_root: str = None,
            projection_dims: int = 1024,
            temperature: float = 1.0,
            weight_decay: float = 0.0,
            lr: float = 1e-3,
            lr_warmup_epochs: int = 5,
            batch_size: int = 64,
            old_checkpoint_path: str = None,
            current_task: int = 0,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = my_load(name=model_name, download_root=download_root)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.val_img_feats = []
        self.val_text_feats = []

        if current_task > 0:
            self.initialize_old_modules()

    def initialize_old_modules(self):
        # load task N-1 checkpoint
        checkpoint = torch.load(self.hparams.old_checkpoint_path, map_location=torch.device('cpu'))
        filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith(
            ('model_old', 'distill_predictor'))}
        self.load_state_dict(filtered_state_dict, strict=False)
        print("Model weights loaded successfully and old parts copied.")

        # Copy model_old
        self.model_old = copy.deepcopy(self.model)
        # Set requires_grad to False for all parameters in the old modules
        for param in self.model_old.parameters():
            param.requires_grad = False

        # distill project
        distill_proj_hidden_dim = 2048
        self.distill_predictor = nn.Sequential(
            nn.Linear(self.hparams.projection_dims, distill_proj_hidden_dim),
            nn.BatchNorm1d(distill_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(distill_proj_hidden_dim, self.hparams.projection_dims),
        )

    def forward(self, inputs):
        image_features = self.model.encode_image(inputs["image"])
        text_features = self.model.encode_text(inputs["caption"])
        return image_features, text_features

    def forward_old(self, inputs):
        with torch.no_grad():
            image_features = self.model_old.encode_image(inputs["image"])
            text_features = self.model_old.encode_text(inputs["caption"])
        return image_features, text_features

    def configure_optimizers(self):
        parameters = [{
            "params": self.model.parameters(),
             "lr": self.hparams.lr,
             "weight_decay": self.hparams.weight_decay
        }]
        if self.hparams.current_task > 0:
            distill_params = [{
                "params": self.distill_predictor.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay,
            }]
            parameters.extend(distill_params)

        optimizer = optim.Adam(parameters, weight_decay=self.hparams.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.lr_warmup_epochs,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=0.01 * self.hparams.lr,
            eta_min=0.01 * self.hparams.lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def _compute_losses_old(self, image_embeddings, text_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.hparams.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.hparams.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (images_loss + texts_loss) / 2.0

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

        logits = torch.einsum("if, jf -> ij", z, z) / self.hparams.temperature
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

    def simclr_distill_loss_func(
            self,
            p1: torch.Tensor,
            p2: torch.Tensor,
            z1: torch.Tensor,
            z2: torch.Tensor,
            # temperature: float = 0.1,
    ) -> torch.Tensor:

        device = self.device

        b = z1.size(0)

        p = F.normalize(torch.cat([p1, p2]), dim=-1)
        z = F.normalize(torch.cat([z1, z2]), dim=-1)

        logits = torch.einsum("if, jf -> ij", p, z) / self.hparams.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
        pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        pos_mask.fill_diagonal_(True)

        # all matches excluding the main diagonal
        logit_mask = torch.ones_like(pos_mask, device=device)
        logit_mask.fill_diagonal_(True)
        logit_mask[:, b:].fill_diagonal_(True)
        logit_mask[b:, :].fill_diagonal_(True)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        # loss
        loss = -mean_log_prob_pos.mean()
        return loss

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        self.log("train/clip_loss", clip_loss, sync_dist=True)

        if self.hparams.current_task > 0:
            frozen_z1, frozen_z2 = self.forward_old(batch)
            p1 = self.distill_predictor(image_embeddings)
            p2 = self.distill_predictor(text_embeddings)

            distill_loss = (
                                   self.simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2)
                                   + self.simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2)
                           ) / 2
            self.log("train/distill_loss", distill_loss, sync_dist=True)
            return clip_loss + distill_loss
        else:
            return clip_loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        self.log("val/clip_loss", clip_loss, sync_dist=True)
        self.val_img_feats.append(image_embeddings)
        self.val_text_feats.append(text_embeddings)

        if self.hparams.current_task > 0:
            frozen_z1, frozen_z2 = self.forward_old(batch)
            p1 = self.distill_predictor(image_embeddings)
            p2 = self.distill_predictor(text_embeddings)

            distill_loss = (
                                   self.simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2)
                                   + self.simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2)
                           ) / 2
            self.log("val/distill_loss", distill_loss, sync_dist=True)
            return clip_loss + distill_loss
        else:
            return clip_loss

    def on_validation_epoch_end(self):
        all_image_features = torch.cat(self.val_img_feats)
        all_text_features = torch.cat(self.val_text_feats)
        val_metrics = self.get_clip_metrics_cpu(
            image_features=all_image_features,
            text_features=all_text_features,
            logit_scale=1 / self.hparams.temperature,
        )
        self.log_dict(val_metrics, sync_dist=True)
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


if __name__ == "__main__":
    from clip_openai.model import my_load
    from dataloaders import ImageRetrievalDataModule

    dm = ImageRetrievalDataModule(
        dataset_name='flickr30k',
        config='../config.yaml',
        root_dir='../artifacts/flickr-30k:v0',
        val_split=0.2,
        max_length=77
    )
    dm.setup(stage='fit')
    a = dm.train_dataloader()
    inputs = next(iter(a))

    model = my_load(name='RN50', download_root='./')
    image_features = model.encode_image(inputs["image"])
    text_features = model.encode_text(inputs["caption"])
    print(image_features.shape)
    print(text_features.shape)
