from lightning.pytorch import cli
from dataloaders.data_module_aug_dil import ImageRetrievalDataModule
from models.clip_model import CLIPDualEncoderModel
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
        parser.add_lightning_class_args(
            LogPredictionCallback, "log_prediction_callback"
        )
        parser.link_arguments(
            "model.text_encoder_alias", "log_prediction_callback.tokenizer"
        )
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")

        # num_tasks
        parser.add_argument(
            "--num_tasks",
            type=int,
            default=1,
            help="Number of incremental learning tasks"
        )
        parser.add_argument(
            "--old_ckpt",
            type=str,
            default=None
        )
        parser.add_argument(
            "--new_ckpt",
            type=str,
            default='clip.ckpt'
        )

        parser.link_arguments(
            "new_ckpt", "model_checkpoint.filename"
        )

    def before_fit(self):
        if self.config.old_ckpt:
            self.model = self.model.load_from_checkpoint(self.config.old_ckpt, strict=True)


if __name__ == "__main__":
    CLI(CLIPDualEncoderModel, ImageRetrievalDataModule, save_config_callback=None)
