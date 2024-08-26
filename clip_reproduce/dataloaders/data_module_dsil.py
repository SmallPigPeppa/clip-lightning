from typing import Optional, List, Union
from torch.utils.data import DataLoader, Subset, ConcatDataset
from lightning import LightningDataModule
from transformers import DistilBertTokenizer
from .flickr30k import Flickr30kDataset
from .coco2014 import COCO2014Dataset
from .cub200 import CUB200Dataset
from .food import UPMCFood101Dataset
from .img_transforms import image_transform_v2

DATASET_LOOKUP = {
    'flickr30k': Flickr30kDataset,
    'coco2014': COCO2014Dataset,
    'cub200': CUB200Dataset,
    'food': UPMCFood101Dataset
}


class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: Union[List[str], str],
            config: Union[List[str], str],
            root_dir: str = None,
            tokenizer_alias: Optional[str] = None,
            max_length: int = 77,
            batch_size: int = 64,
            num_workers: int = 8,
            num_tasks: int = 1,
            current_task: int = 0,
            pin_memory: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # If dataset_name is a str, convert it to a list with a single element
        if isinstance(dataset_name, str):
            self.dataset_name = [dataset_name]
        else:
            self.dataset_name = dataset_name

        # If config is a str, expand it to match the length of dataset_name
        if isinstance(config, str):
            self.config = [config] * len(self.dataset_name)
        else:
            self.config = config

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_alias)
        self.max_length = max_length
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.current_task = current_task
        self.pin_memory = pin_memory

    def setup(
            self,
            stage: Optional[str] = None,
    ) -> None:
        self.train_datasets = []
        self.val_datasets = []

        # Create train and val datasets for each dataset_name and config pair
        for name, cfg in zip(self.dataset_name, self.config):
            train_dataset = DATASET_LOOKUP[name](
                root_dir=self.root_dir,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                split='train',
                transforms=image_transform_v2(config_path=cfg, is_train=True)
            )
            val_dataset = DATASET_LOOKUP[name](
                root_dir=self.root_dir,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                split='val',
                transforms=image_transform_v2(config_path=cfg, is_train=False)
            )
            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)

        self.task_datasets = self.train_datasets

    def train_dataloader(self):
        return DataLoader(
            self.task_datasets[self.current_task],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        # Create individual validation loaders for each dataset
        val_loaders = []
        for val_dataset in self.val_datasets:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            val_loaders.append(val_loader)

        # Create a combined validation loader for all datasets
        combined_val_dataset = ConcatDataset(self.val_datasets)
        combined_val_loader = DataLoader(
            combined_val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        # Return a list of all individual loaders followed by the combined loader
        return val_loaders + [combined_val_loader]
