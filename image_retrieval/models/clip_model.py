import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
            lr_warmup_epochs: int = 5,
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
        self.lr_warmup_epochs = lr_warmup_epochs

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
        # optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        optimizer = optim.SGD(parameters, weight_decay=5e-4, momentum=0.9)
        base_lr = min(self.image_encoder_lr, self.text_encoder_lr, self.head_lr)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.lr_warmup_epochs,
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
