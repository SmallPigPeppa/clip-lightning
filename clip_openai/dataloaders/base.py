from abc import abstractmethod
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageRetrievalDataset(Dataset):
    def __init__(
            self, root_dir, tokenizer, max_length: int = 200, transforms=None
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.images, self.captions = self.fetch_dataset()

    def __len__(self):
        return len(self.captions)

    def tokenize(self, text):
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        a = self.tokenizer.encode(text)
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.max_length, dtype=torch.long)
        if len(tokens) <= self.max_length:
            result[:len(tokens)] = torch.tensor(tokens)
        else:
            result[:self.max_length] = torch.tensor(tokens)[:self.max_length]
        return result

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        caption = self.tokenize(self.captions[index])
        if self.transforms:
            image = self.transforms(image)

        return {"image": image, "caption": caption}

    @abstractmethod
    def fetch_dataset(self):
        pass
