import random
import torch
from packaging import version
from torch.utils.data import Dataset
from datasets import load_dataset


# Define dataset mappings as a constant outside the class
DATASET_MAPPINGS = {
    'emoji': {
        'hf_name': 'Norod78/microsoft-fluentui-emoji-512-whitebg',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': None
    },
    'wikiart': {
        'hf_name': 'sAterMors/wikiart_recaption',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': None
    },
    'face': {
        'hf_name': 'OpenFace-CQUPT/FaceCaption-15M',
        'keys': {'image': 'image', 'text': 'caption'},
        'splits': None
    },
    'newyorker': {
        'hf_name': 'jmhessel/newyorker_caption_contest',
        'keys': {'image': 'image', 'text': 'image_description'},
        'splits': {'train': 'train', 'val': 'validation'}
    },
    'patfig': {
        'hf_name': 'lcolonn/patfig',
        'keys': {'image': 'image', 'text': 'short_description'},
        'splits': {'train': 'train', 'val': 'test'}
    },
    'fashion': {
        'hf_name': 'jinaai/fashion-captions-de',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 'train', 'val': 'test'}
    },
    'pet': {
        'hf_name': 'visual-layer/oxford-iiit-pet-vl-enriched',
        'keys': {'image': 'image', 'text': 'caption_enriched'},
        'splits': {'train': 'train', 'val': 'test'}
    },
}


class ImageRetrievalDataset(Dataset):
    def __init__(
            self,
            dataset_name,
            root_dir,
            tokenizer,
            max_length: int = 200,
            transforms=None, split='train'
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Get the dataset information from the constant
        dataset_info = DATASET_MAPPINGS.get(dataset_name)
        if dataset_info is None:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        # Handle split mapping
        self.split = dataset_info['splits'].get(split, split)  # Default to the provided split if not found

        # Load the dataset with the correct split
        self.hf_dataset = load_dataset(dataset_info['hf_name'], split=self.split)

        # Save the keys for later use
        self.keys = dataset_info['keys']

    def __len__(self):
        return len(self.hf_dataset)

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
    #     sample = self.hf_dataset[index]
    #
    #     # Extract image and text using the keys from the dictionary
    #     image = sample[self.keys['image']]
    #     text = sample[self.keys['text']]

    def __getitem__(self, index):
        # image = Image.open(self.images[index])
        # caption = self.captions[index]
        sample = self.hf_dataset[index]

        # Extract image and text using the keys from the dictionary
        image = sample[self.keys['image']]
        caption = sample[self.keys['text']]
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = self.tokenize(caption)
        if self.transforms:
            image = self.transforms(image)

        return {"image": image, "caption": caption}

