from lightning.pytorch import cli
from dataloaders.data_module_dil import ImageRetrievalDataModule
from model_reproduce.clip_model_simclr import CLIPDualEncoderModel
from callbacks import LogPredictionCallback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

import os

os.environ['CURL_CA_BUNDLE'] = ''


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("model.text_encoder_alias", "data.tokenizer_alias")
        parser.link_arguments(
            "data.train_batch_size", "model.train_batch_size"
        )
        parser.link_arguments(
            "data.val_batch_size", "model.val_batch_size"
        )

        # log prediction
        # parser.add_lightning_class_args(
        #     LogPredictionCallback, "log_prediction_callback"
        # )
        # parser.link_arguments(
        #     "model.text_encoder_alias", "log_prediction_callback.tokenizer"
        # )
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")


if __name__ == "__main__":
    CLI(CLIPDualEncoderModel, ImageRetrievalDataModule, save_config_callback=None, seed_everything_default=6)
