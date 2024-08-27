from typing import Optional
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Subset
from lightning import LightningDataModule
from .base import ImageRetrievalDataset
from .flickr30k_json import Flickr30kDataset
from .img_transforms import image_transform_v2
from model_openai import SimpleTokenizer
from .cub200 import CUB200Dataset
from .food import UPMCFood101Dataset

DATASET_LOOKUP = {
    'flickr30k': Flickr30kDataset,
    'cub200': CUB200Dataset,
    'food': UPMCFood101Dataset
}


class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            config: str,
            root_dir: str = None,
            max_length: int = 77,
            batch_size: int = 64,
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
        self.tokenizer = SimpleTokenizer()
        self.max_length = max_length
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.current_task = current_task

    @staticmethod
    def split_data(dataset: ImageRetrievalDataset, val_split: float):
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

        # 划分训练集为多个任务
        task_size = len(train_dataset) // self.num_tasks
        self.task_datasets = [Subset(train_dataset, range(i * task_size, (i + 1) * task_size)) for i in
                              range(self.num_tasks)]

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
