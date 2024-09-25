from abc import abstractmethod
import torch
from PIL import Image
from torch.utils.data import Dataset
from packaging import version
from typing import Union, List
import random


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
        self.caption_num = 1

    @abstractmethod
    def fetch_dataset(self, split):
        pass

    def __len__(self):
        return len(self.captions)

    def tokenize(self, text):
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        if version.parse(torch.__version__) < version.parse("1.8.0"):
            result = torch.zeros(self.max_length, dtype=torch.long)
        else:
            result = torch.zeros(self.max_length, dtype=torch.int)

        if len(tokens) <= self.max_length:
            result[:len(tokens)] = torch.tensor(tokens)
        else:
            result[:self.max_length] = torch.tensor(tokens)[:self.max_length]
        return result

    def __getitem__(self, index):
        image_i = Image.open(self.images[index])
        if self.caption_num > 1:
            caption_i = self.captions[index][:self.caption_num]
        else:
            caption_i = self.captions[index]
        if self.transforms:
            image_i = self.transforms(image_i)

        return {"image": image_i, "text": caption_i}
