from abc import abstractmethod
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
from datasets import load_dataset


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
        self.hf_dataset = load_dataset("Norod78/microsoft-fluentui-emoji-512-whitebg", split=split)


    def __len__(self):
        return len(self.hf_dataset)

    def tokenize(self, text):
        result = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_length
        )
        return result

    def __getitem__(self, index):

        # image = Image.open(self.images[index])
        # caption = self.captions[index]
        sample = self.hf_dataset[index]
        image = sample['image']
        caption = sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        item = self.tokenize(caption)
        for key in item.keys():
            item[key] = torch.tensor(item[key])
        if self.transforms:
            image = self.transforms(image)
        item['image'] = image
        return item
