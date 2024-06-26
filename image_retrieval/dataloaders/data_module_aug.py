from typing import Optional
from torch.utils.data import random_split, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from lightning import LightningDataModule
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

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = DATASET_LOOKUP[self.dataset_name](
            artifact_id=self.artifact_id,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        train_transforms = image_transform_v2(config_path=self.config, is_train=True)
        val_transforms = image_transform_v2(config_path=self.config, is_train=False)

        train_length = int((1 - self.val_split) * len(dataset))
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(dataset, lengths=[train_length, val_length])
        train_dataset.dataset.transforms = train_transforms
        val_dataset.dataset.transforms = val_transforms

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size,
            num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size,
            num_workers=self.num_workers, pin_memory=True
        )
