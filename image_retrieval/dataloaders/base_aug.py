from abc import abstractmethod
import torch
from PIL import Image
from torch.utils.data import Dataset
from img_transforms import image_transform_v2


class ImageRetrievalDataset(Dataset):
    def __init__(
            self,
            artifact_id: str,
            config: str,
            is_train: bool,
            tokenizer=None,
            max_length: int = 200,

    ) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.image_files, self.captions = self.fetch_dataset()
        self.images = self.image_files
        assert tokenizer is not None
        self.tokenized_captions = tokenizer(
            list(self.captions), padding=True, truncation=True, max_length=max_length
        )
        self.transforms = image_transform_v2(config_path=config, is_train=is_train, )

    @abstractmethod
    def fetch_dataset(self):
        pass

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index])
            for key, values in self.tokenized_captions.items()
        }
        image = Image.open(self.image_files[index])
        item["image"] = self.transforms(image)
        item["caption"] = self.captions[index]
        return item
