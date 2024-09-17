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

DATASET_LOOKUP = {
    'flickr30k': Flickr30kDataset,
    'coco2014': COCO2014Dataset,
    'cub200': CUB200Dataset,
    'food': UPMCFood101Dataset,
}


class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
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
        self.dataset_name = dataset_name
        self.config = config
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.batch_size_zs = batch_size_zs
        self.tokenizer = SimpleTokenizer()
        self.max_length = max_length
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.current_task = current_task

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
        if self.dataset_name in DATASET_LOOKUP:
            train_dataset = DATASET_LOOKUP[self.dataset_name](
                root_dir=self.root_dir,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                split="train",
                transforms=image_transform_v2(config_path=self.config, is_train=True)
            )
            self.val_dataset = DATASET_LOOKUP[self.dataset_name](
                root_dir=self.root_dir,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                split="val",
                transforms=image_transform_v2(config_path=self.config, is_train=False)
            )
        else:
            dataset_config = DATASET_MAPPINGS[self.dataset_name]

            # 获取数据增强的配置
            train_transforms = image_transform_v2(config_path=self.config, is_train=True)
            val_transforms = image_transform_v2(config_path=self.config, is_train=False)

            if isinstance(dataset_config['splits']['train'], (int, float)):
                # 创建数据集实例（无分割信息）
                full_dataset = ImageRetrievalDatasetHF(
                    dataset_name=self.dataset_name,
                    root_dir=self.root_dir,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    transforms=train_transforms  # 使用训练集变换初始化
                )

                # 如果划分方式为数字比例
                train_ratio = dataset_config['splits']['train']
                val_ratio = dataset_config['splits']['val']
                total_len = len(full_dataset)

                # 计算划分长度
                train_len = int(total_len * train_ratio)
                val_len = int(total_len * val_ratio)  # 确保验证集按自身比例计算

                # 创建一个随机索引列表
                indices = np.arange(total_len)
                np.random.shuffle(indices)

                # 根据索引划分数据集
                train_indices = indices[:train_len].tolist()  # 转换为 Python 列表
                val_indices = indices[train_len:train_len + val_len].tolist()

                # 创建子集
                train_dataset = Subset(full_dataset, train_indices)
                self.val_dataset = Subset(full_dataset, val_indices)

                # 为验证集设置正确的变换
                self.val_dataset.dataset.transforms = val_transforms

            else:
                # 使用预定义的分割
                train_dataset = ImageRetrievalDatasetHF(
                    dataset_name=self.dataset_name,
                    root_dir=self.root_dir,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    split='train',
                    transforms=train_transforms
                )
                self.val_dataset = ImageRetrievalDatasetHF(
                    dataset_name=self.dataset_name,
                    root_dir=self.root_dir,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    split='val',
                    transforms=val_transforms
                )

        # 划分训练集为多个任务
        task_size = len(train_dataset) // self.num_tasks
        self.task_datasets = [
            Subset(train_dataset, range(i * task_size, (i + 1) * task_size))
            for i in range(self.num_tasks)
        ]

        # IMNET zero-shot dataset
        args = argparse.Namespace(
            data_set='IMNET',  # Specify ImageNet dataset
            data_path=os.path.join(self.root_dir, 'imagenet'),  # Path to your ImageNet dataset
            input_size=224,  # Example of image size, adjust according to your needs
            eval_crop_ratio=0.875  # Example of crop percentage, adjust according to your needs
            # Add other relevant parameters as needed
        )

        # Call build_dataset with ImageNet settings
        self.zero_shot_dataset, nb_classes = build_dataset(is_train=False, args=args)

    def train_dataloader(self):
        return DataLoader(
            self.task_datasets[self.current_task],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def zero_shot_dataloader(self):
        return DataLoader(
            self.zero_shot_dataset,
            batch_size=self.batch_size_zs,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":
    dm = ImageRetrievalDataModule(
        dataset_name='flickr30k',
        config='../../config.yaml',
        root_dir='../../artifacts/flickr-30k:v0',
        val_split=0.2,
        max_length=100
    )
    dm.setup(stage='fit')
    a = dm.train_dataloader()
    print(next(iter(a)))
