from typing import Optional
from torch.utils.data import DataLoader, Subset
from lightning import LightningDataModule
from transformers import DistilBertTokenizer
from .flickr30k import Flickr30kDataset
from .coco2014 import COCO2014Dataset
from .img_transforms import image_transform_v2

DATASET_LOOKUP = {
    "flickr30k": Flickr30kDataset,
    "coco2014": COCO2014Dataset
}


class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            config: str,
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
        self.dataset_name = dataset_name
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
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
