from lightning.pytorch import cli
from dataloaders_multicaption.data_module_val import ImageRetrievalDataModule
from model_val_iclr import CLIPDualEncoderModel

import os
os.environ['CURL_CA_BUNDLE'] = ''


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.batch_size_zs", "model.batch_size_zs")



if __name__ == "__main__":
    CLI(
        CLIPDualEncoderModel,
        ImageRetrievalDataModule,
        save_config_callback=None,
        seed_everything_default=6
    )

