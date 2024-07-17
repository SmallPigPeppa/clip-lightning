from typing import Optional
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Subset
from lightning import LightningDataModule
from transformers import DistilBertTokenizer
from .base_aug import ImageRetrievalDataset
from .flickr30k_aug import Flickr30kDataseAug
from .img_transforms import image_transform_v2

DATASET_LOOKUP = {"flickr30k_aug": Flickr30kDataseAug}

class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
            self,
            artifact_id: str,
            dataset_name: str,
            config: str,
            val_split: float = 0.2,
            tokenizer_alias: Optional[str] = None,
            max_length: int = 100,
            train_batch_size: int = 16,
            val_batch_size: int = 16,
            num_workers: int = 8,
            num_tasks: int = 1,
            current_task: int = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.artifact_id = artifact_id
        self.dataset_name = dataset_name
        self.config = config
        self.val_split = val_split
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_alias)
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
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
        dataset = DATASET_LOOKUP[self.dataset_name](
            artifact_id=self.artifact_id,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            transforms=transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        )
        train_dataset, self.val_dataset = self.split_data(dataset, val_split=self.val_split)

        # 划分训练集为多个任务
        task_size = len(train_dataset) // self.num_tasks
        self.task_datasets = [Subset(train_dataset, range(i * task_size, (i + 1) * task_size)) for i in range(self.num_tasks)]

        # 对每个任务的数据集应用不同的变换
        train_transforms = image_transform_v2(config_path=self.config, is_train=True)
        val_transforms = image_transform_v2(config_path=self.config, is_train=False)

        for task_dataset in self.task_datasets:
            task_dataset.dataset.transforms = train_transforms
        self.val_dataset.dataset.transforms = val_transforms

    def train_dataloader(self):
        return DataLoader(
            self.task_datasets[self.current_task],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
