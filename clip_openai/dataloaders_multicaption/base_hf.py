import torch
from packaging import version
from torch.utils.data import Dataset
from datasets import load_dataset
import requests
from PIL import Image
import os

# Define dataset mappings as a constant outside the class
DATASET_MAPPINGS = {
    'wikiart': {
        'hf_name': 'AterMors/wikiart_recaption',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 0.8, 'val': 0.2}
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
    'shahnegar': {
        'hf_name': 'sadrasabouri/ShahNegar',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 0.8, 'val': 0.2}
    },
    'artbench': {
        'hf_name': 'alfredplpl/artbench-pd-256x256',
        'keys': {'image': 'image', 'text': 'caption'},
        'splits': {'train': 0.8, 'val': 0.2}
    },
    'hausavg': {
        'hf_name': 'HausaNLP/HausaVG',
        'keys': {'image': 'image', 'text': 'en_text'},
        'splits': {'train': 'train', 'val': 'validation'}
    },
    'simpsons': {
        'hf_name': 'bigdata-pw/TheSimpsons',
        'keys': {'image': 'jpg', 'text': 'caption.txt'},
        'splits': {'train': 0.8, 'val': 0.2}
    },
    'lexica': {
        'hf_name': 'vera365/lexica_dataset',
        'keys': {'image': 'image', 'text': 'prompt'},
        'splits': {'train': 'train', 'val': 'test'}
    },
    'styles': {
        'hf_name': 'rezashkv/styles',
        'keys': {'image': 'image', 'text': 'caption'},
        'splits': {'train': 0.5, 'val': 0.5}
    },
    'kream': {
        'hf_name': 'hahminlew/kream-product-blip-captions',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 0.5, 'val': 0.5}
    },
    'sketch': {
        'hf_name': 'zoheb/sketch-scene',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 0.8, 'val': 0.2}
    },
    'clothes': {
        'hf_name': 'Luna288/image-captioning-FACAD-base',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 0.8, 'val': 0.2}
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


        if isinstance(dataset_info['splits']['train'], (int, float)):
            self.split = 'train'
        else:
            self.split = dataset_info['splits'].get(split, split)

        # Load the dataset with the correct split
        self.hf_dataset = load_dataset(dataset_info['hf_name'], split=self.split)

        # Save the keys for later use
        self.keys = dataset_info['keys']

        self.caption_num = 1

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


    def download_image(self, url):
        # Modify the directory structure to include '/cache'
        cache_dir = os.path.join(self.root_dir, 'cache')
        file_path = os.path.join(cache_dir, *url.split('/')[2:])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Download the image if it doesn't exist locally
        if not os.path.exists(file_path):
            # response = requests.get(url)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
            }

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                # Print error and return None if the download fails
                print(f"Failed to download {url}")
                return None  # Return None to indicate failure
        return file_path

    def __getitem__(self, index):
        sample = self.hf_dataset[index]
        image_url = sample[self.keys['image']]
        text = sample[self.keys['text']]

        # Check if 'url' is in the key for image data to determine if it's a URL
        if 'url' in self.keys['image'] or 'URL' in self.keys['image']:
            image_path = self.download_image(image_url)
            if image_path is None:
                # If image_path is None, access the 0th element
                sample = self.hf_dataset[0]
                image_url = sample[self.keys['image']]
                image_path = self.download_image(image_url)  # Attempt to download image from the 0th element
                # Update text from the 0th element since image download was successful
                text = sample[self.keys['text']]

            image = Image.open(image_path)

        else:  # Direct image loading without download
            image = sample[self.keys['image']]

        if isinstance(text, list) and len(text) == 1:
            text = text[0]
        elif isinstance(text, str):
            pass
        else:
            import warnings
            warnings.warn(f"self.dataset_name is {self.dataset_name}: The text list contains {len(text)} elements.")
            longest_text = max(text, key=len)
            text = longest_text


        if self.transforms:
            image = self.transforms(image)

        return {"image": image, "text": text}
