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

    # def __getitem__(self, index):
    #     image = Image.open(self.images[index])
    #     caption = self.captions[index]
    #     if isinstance(caption, list):
    #         caption = random.choice(caption)
    #     caption = self.tokenize(caption)
    #     if self.transforms:
    #         image = self.transforms(image)
    #
    #     return {"image": image, "caption": caption}

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        caption = self.captions[index]

        if isinstance(caption, list):
            if len(caption) == 1:
                caption = self.tokenize(caption[0])  # Only one element, return the original
                multi_caption = False
            else:
                caption = [self.tokenize(c) for c in caption]  # Tokenize all captions if more than one
                multi_caption = True
        else:
            caption = self.tokenize(caption)
            multi_caption = False  # Not a list, so it's not multi-caption

        if self.transforms:
            image = self.transforms(image)

        return {"image": image, "caption": caption, "multi_caption": multi_caption}



