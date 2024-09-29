from typing import Optional
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from dataset_others import get_dataset


class ZeroshotDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            config: str,
            root_dir: str = None,
            batch_size: int = 32,
            num_workers: int = 8,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.dataset_name = dataset_name
        self.dataset_name = dataset_name.split(",")  # Split the string into a list
        self.config = config
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Store datasets for later use
        self.datasets = {}

    def setup_single_dataset(
            self,
            dataset_name: str,
            stage: Optional[str] = None,
    ):
        val_dataset = get_dataset(dataset_name=dataset_name, data_path=self.root_dir)
        # Store datasets for each dataset name
        self.datasets[dataset_name] = {
            "val": val_dataset
        }

    def setup(self, stage: Optional[str] = None) -> None:
        # Loop over all dataset names and call the setup function for each one
        for dataset_name in self.dataset_name:
            self.setup_single_dataset(dataset_name, stage)

    def zero_shot_dataloader(self, dataset_name=None):
        return DataLoader(
            self.datasets[dataset_name]["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self, dataset_name=None):
        dataset_name = dataset_name or self.dataset_name[0]  # Default to the first dataset if none is provided
        return DataLoader(
            self.datasets[dataset_name]["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
