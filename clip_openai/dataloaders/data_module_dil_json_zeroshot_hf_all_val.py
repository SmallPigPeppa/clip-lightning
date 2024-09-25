
from typing import Optional
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Subset
from lightning import LightningDataModule
from .img_transforms import image_transform_v2
from model_openai import SimpleTokenizer

from .imagenet import build_dataset
import argparse
import os
from .base_hf import ImageRetrievalDataset as ImageRetrievalDatasetHF
from .base_hf import DATASET_MAPPINGS
from .base import ImageRetrievalDataset
from .flickr30k_json import Flickr30kDataset
from .cub200 import CUB200Dataset
from .food import UPMCFood101Dataset
from .coco2014 import COCO2014Dataset

import torch
import numpy as np
from typing import List


DATASET_LOOKUP = {
    'flickr30k': Flickr30kDataset,
    'coco2014': COCO2014Dataset,
    'cub200': CUB200Dataset,
    'food': UPMCFood101Dataset,
}





class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_names: List[str],  # Modify to accept a list of dataset names
            config: str,
            root_dir: str = None,
            max_length: int = 77,
            batch_size: int = 64,
            batch_size_zs: int = 256,
            num_workers: int = 8,
            num_tasks: int = 1,
            current_task: int = 0,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_names = dataset_names  # Store the list of dataset names
        self.config = config
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.batch_size_zs = batch_size_zs
        self.tokenizer = SimpleTokenizer()
        self.max_length = max_length
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.current_task = current_task

        self.datasets = {}
        self.val_datasets = {}

    @staticmethod
    def split_data(dataset: ImageRetrievalDatasetHF, val_split: float):
        train_length = int((1 - val_split) * len(dataset))
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(
            dataset, lengths=[train_length, val_length]
        )
        return train_dataset, val_dataset

    def setup(
            self,
            stage: Optional[str] = None,
    ) -> None:
        # Loop through all dataset names to create train and val datasets
        for dataset_name in self.dataset_names:
            if dataset_name in DATASET_LOOKUP:
                train_dataset = DATASET_LOOKUP[dataset_name](
                    root_dir=self.root_dir,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    split="train",
                    transforms=image_transform_v2(config_path=self.config, is_train=True)
                )
                val_dataset = DATASET_LOOKUP[dataset_name](
                    root_dir=self.root_dir,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    split="val",
                    transforms=image_transform_v2(config_path=self.config, is_train=False)
                )
            else:
                dataset_config = DATASET_MAPPINGS[dataset_name]

                train_transforms = image_transform_v2(config_path=self.config, is_train=True)
                val_transforms = image_transform_v2(config_path=self.config, is_train=False)

                full_dataset = ImageRetrievalDatasetHF(
                    dataset_name=dataset_name,
                    root_dir=self.root_dir,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    transforms=train_transforms
                )

                train_len = int(len(full_dataset) * dataset_config['splits']['train'])
                val_len = len(full_dataset) - train_len

                indices = np.arange(len(full_dataset))
                np.random.shuffle(indices)

                train_indices = indices[:train_len].tolist()
                val_indices = indices[train_len:].tolist()

                train_dataset = Subset(full_dataset, train_indices)
                val_dataset = Subset(full_dataset, val_indices)
                val_dataset.dataset.transforms = val_transforms  # Apply val transforms

            self.datasets[dataset_name] = train_dataset
            self.val_datasets[dataset_name] = val_dataset

        # IMNET zero-shot dataset (only for the first dataset)
        args = argparse.Namespace(
            data_set='IMNET',
            data_path=os.path.join(self.root_dir, 'imagenet'),
            input_size=224,
            eval_crop_ratio=0.875
        )
        self.zero_shot_dataset, _ = build_dataset(is_train=False, args=args)

    def train_dataloader(self, dataset_name=None):
        dataset_name = dataset_name or self.dataset_names[0]  # Default to the first dataset
        return DataLoader(
            self.datasets[dataset_name],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self, dataset_name=None):
        if dataset_name:
            return DataLoader(
                self.val_datasets[dataset_name],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            # Create a dictionary of dataloaders for each dataset
            return {name: DataLoader(self.val_datasets[name], batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
                    for name in self.dataset_names}

    def zero_shot_dataloader(self):
        return DataLoader(
            self.zero_shot_dataset,
            batch_size=self.batch_size_zs,
            num_workers=self.num_workers,
            pin_memory=True
        )
