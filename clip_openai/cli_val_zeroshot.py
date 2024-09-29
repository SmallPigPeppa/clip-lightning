from lightning.pytorch import cli
from zeroshot_others.data_module import ZeroshotDataModule
from model_val_zeroshot import CLIPDualEncoderModel

import os
os.environ['CURL_CA_BUNDLE'] = ''


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser: cli.LightningArgumentParser) -> None:
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.batch_size_zs", "model.batch_size_zs")



if __name__ == "__main__":
    CLI(
        CLIPDualEncoderModel,
        ZeroshotDataModule,
        save_config_callback=None,
        seed_everything_default=6
    )

