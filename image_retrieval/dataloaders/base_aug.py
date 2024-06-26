from abc import abstractmethod
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageRetrievalDataset(Dataset):
    def __init__(
            self, artifact_id: str, tokenizer, max_length: int = 200, transforms=None
    ) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.image_files, self.captions = self.fetch_dataset()
        self.transforms = transforms
        self.tokenized_captions = tokenizer(
            list(self.captions), padding=True, truncation=True, max_length=max_length
        )

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
        if self.transforms:
            image = self.transforms(image)
        item["image"] = image
        item["caption"] = self.captions[index]
        return item