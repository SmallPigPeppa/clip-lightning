from abc import abstractmethod
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageRetrievalDataset(Dataset):
    def __init__(
            self, root_dir, tokenizer, max_length: int = 200, transforms=None, split='train'
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.images, self.captions = self.fetch_dataset(split=split)

    @abstractmethod
    def fetch_dataset(self, split):
        pass

    def __len__(self):
        return len(self.captions)

    def tokenize(self, text):
        result = self.tokenizer(
            text, padding=True, truncation=True, max_length=self.max_length
        )
        return result

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
