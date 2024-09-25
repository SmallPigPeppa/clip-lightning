from lightning.pytorch import cli
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from dataloaders.data_module_dil_json_zeroshot_hf_all_val import ImageRetrievalDataModule
from model_val import CLIPDualEncoderModel

import os
os.environ['CURL_CA_BUNDLE'] = ''


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.batch_size_zs", "model.batch_size_zs")
        parser.link_arguments("data.current_task", "model.current_task")
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")

        # Accept multiple datasets from the command line
        parser.add_argument(
            "--data.datasets",
            type=str,
            nargs='+',  # allow multiple datasets to be passed as a list
            help="List of datasets to evaluate on"
        )


if __name__ == "__main__":
    CLI(
        CLIPDualEncoderModel,
        ImageRetrievalDataModule,
        save_config_callback=None,
        seed_everything_default=6
    )

