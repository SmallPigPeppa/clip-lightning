from typing import Optional
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from lightning import LightningDataModule
from .base import ImageRetrievalDataset
from .flickr30k import Flickr30kDataset
from .img_transforms import image_transform_v2
from ..model_openai import SimpleTokenizer

DATASET_LOOKUP = {"flickr30k": Flickr30kDataset}


class ImageRetrievalDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_name: str,
            config: str,
            val_split: float = 0.2,
            root_dir: str = None,
            max_length: int = 100,
            train_batch_size: int = 16,
            val_batch_size: int = 16,
            num_workers: int = 8,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.config = config
        self.root_dir = root_dir
        self.val_split = val_split
        self.tokenizer = SimpleTokenizer()
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

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
            root_dir=self.root_dir,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            transforms=transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        )
        self.train_dataset, self.val_dataset = self.split_data(
            dataset, val_split=self.val_split
        )

        train_transforms = image_transform_v2(config_path=self.config, is_train=True)
        val_transforms = image_transform_v2(config_path=self.config, is_train=False)

        self.train_dataset.transforms = train_transforms
        self.val_dataset.transforms = val_transforms

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
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
