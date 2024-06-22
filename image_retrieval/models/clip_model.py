import itertools

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from pytorch_lightning import LightningModule
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from .encoders import ImageEncoder, ProjectionHead, TextEncoder


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
            lr_scheduler_patience: float = 1.0,
            lr_scheduler_factor: float = 0.8,
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
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.image_encoder_lr = image_encoder_lr
        self.text_encoder_lr = text_encoder_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor

        self.save_hyperparameters()

    def _compute_losses(self, image_embeddings, text_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (images_loss + texts_loss) / 2.0

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
            {"params": self.image_encoder.parameters(), "lr": self.image_encoder_lr},
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {
                "params": itertools.chain(
                    self.image_projection.parameters(),
                    self.text_projection.parameters(),
                ),
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     patience=self.lr_scheduler_patience,
        #     factor=self.lr_scheduler_factor,
        # )
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=5,
        #     max_epochs=max_epochs,
        #     warmup_start_lr=0.01 * lr,
        #     eta_min=0.01 * lr,
        # )

        # 选择最小的学习率作为warmup和annealing的参考基线
        base_lr = min(self.image_encoder_lr, self.text_encoder_lr, self.head_lr)

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=self.max_epochs,
            warmup_start_lr=0.01 * base_lr,  # 基于最小学习率计算起始学习率
            eta_min=0.01 * base_lr  # 设置最小学习率
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        train_loss = self.all_gather(loss)
        self.log("train/loss", train_loss.mean())
        return loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        val_loss = self.all_gather(loss)
        self.log("val/loss", val_loss.mean())
        return loss
