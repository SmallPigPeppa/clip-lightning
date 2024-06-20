# from pytorch_lightning.utilities import cli
# import pytorch_lightning.cli.LightningCLI as CIL
# from pytorch_lightning import cli
from lightning.pytorch import cli

from dataloaders.data_module import ImageRetrievalDataModule
from models.clip_model import CLIPDualEncoderModel
from callbacks import LogPredictionCallback
# from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

import os

os.environ['CURL_CA_BUNDLE'] = ''


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("model.text_encoder_alias", "data.tokenizer_alias")
        parser.add_lightning_class_args(
            LogPredictionCallback, "log_prediction_callback"
        )
        parser.link_arguments(
            "model.text_encoder_alias", "log_prediction_callback.tokenizer"
        )
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")


if __name__ == "__main__":
    # CLI(CLIPDualEncoderModel, ImageRetrievalDataModule, save_config_callback=None)
    '''
    python image_retrieval/cli.py fit \
    --data.dataset_name flickr30k \
    --data.artifact_id wandb/clip.lightning-image_retrieval/flickr-30k:latest \
    --data.train_batch_size 128 \
    --data.val_batch_size 128 \
    --data.max_length 200 \
    --model.image_encoder_alias resnet50 \
    --model.text_encoder_alias distilbert-base-uncased \
    --model.image_embedding_dims 2048 \
    --model.text_embedding_dims 768 \
    --model.projection_dims 256 \
    --trainer.precision 16 \
    --trainer.accelerator gpu \
    --trainer.max_epochs 20 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger
    '''
    mdata = ImageRetrievalDataModule(
        dataset_name='flickr30k',
        artifact_id='wandb/clip.lightning-image_retrieval/flickr-30k:latest',
        train_batch_size=128,
        val_batch_size=128,
        max_length=200,
        tokenizer_alias='distilbert-base-uncased', )

    model = CLIPDualEncoderModel(
        image_encoder_alias='resnet50',
        text_encoder_alias='distilbert-base-uncased',
        image_embedding_dims=2048,
        text_embedding_dims=768,
        projection_dims=256, )
    mdata.setup()
    train_loader = mdata.train_dataloader()
    batch = next(iter(train_loader))
    loss = model.train_step(batch)
