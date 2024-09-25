import random
import torch
from packaging import version
from torch.utils.data import Dataset
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import os

# Define dataset mappings as a constant outside the class
DATASET_MAPPINGS = {
    'emoji': {
        'hf_name': 'Norod78/microsoft-fluentui-emoji-512-whitebg',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 0.5, 'val': 0.5}
    },
    'wikiart': {
        'hf_name': 'AterMors/wikiart_recaption',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 0.8, 'val': 0.2}
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
    'nouns': {
        'hf_name': 'm1guelpf/nouns',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 0.5, 'val': 0.5}
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
    'wit': {
        'hf_name': 'wikimedia/wit_base',
        'keys': {'image': 'image', 'text': 'caption_attribution_description'},
        'splits': {'train': 0.8, 'val': 0.2}
    },
    'oldbook': {
        'hf_name': 'gigant/oldbookillustrations',
        'keys': {'image': 'rawscan', 'text': 'image_caption'},
        'splits': {'train': 0.8, 'val': 0.2}
    },
    'polaris': {
        'hf_name': 'yuwd/Polaris',
        'keys': {'image': 'img', 'text': 'refs'},
        'splits': {'train': 'train', 'val': 'test'}
    },
    'news': {
        'hf_name': 'Oztobuzz/Kosmos_news',
        'keys': {'image': 'img', 'text': 'caption'},
        'splits': {'train': 0.8, 'val': 0.2}
    },
    'cxiu': {
        'hf_name': 'Shrey-1329/cxiu_hf_dataset',
        'keys': {'image': 'image', 'text': 'text'},
        'splits': {'train': 0.8, 'val': 0.2}
    },

    # 'text2food': {
    #     'hf_name': 'tum-nlp/text2food-mmc4',
    #     'keys': {'image': 'Raw URL', 'text': 'Matched Text'},
    #     'splits': {'train': 0.8, 'val': 0.2}
    # },
    # 'plans': {
    #     'hf_name': 'ShazShoaib/SingleFloorPlans',
    #     'keys': {'image': 'image', 'text': 'text'},
    #     'splits': {'train': 0.8, 'val': 0.2}
    # },
    # 'tomato': {
    #     'hf_name': 'wellCh4n/tomato-leaf-disease-image',
    #     'keys': {'image': 'image', 'text': 'text'},
    #     'splits': {'train': 0.8, 'val': 0.2}
    # },
    # 'peanuts': {
    #     'hf_name': 'afmck/peanuts-flan-t5-xl',
    #     'keys': {'image': 'image', 'text': 'caption'},
    #     'splits': {'train': 0.8, 'val': 0.2}
    # },
    # 'vintage': {
    #     'hf_name': 'SilentAntagonist/vintage-artworks-60k-captioned',
    #     'keys': {'image': 'image_url', 'text': 'short_caption'},
    #     'splits': {'train': 0.5, 'val': 0.5}
    # },
    # 'face': {
    #     'hf_name': 'OpenFace-CQUPT/FaceCaption-15M',
    #     'keys': {'image': 'url', 'text': 'caption'},
    #     'splits': {'train': 0.002, 'val': 0.0005}
    # },
    # 'pokemon': {
    #     'hf_name': 'TheFusion21/PokemonCards',
    #     'keys': {'image': 'image_url', 'text': 'caption'},
    #     'splits': {'train': 0.5, 'val': 0.5}
    # },
    # 'midjourney': {
    #     'hf_name': 'CortexLM/midjourney-v6',
    #     'keys': {'image': 'image_url', 'text': 'prompt'},
    #     'splits': {'train': 0.5, 'val': 0.5}
    # },

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

        # # Handle split mapping
        # self.split = dataset_info['splits'].get(split, split)  # Default to the provided split if not found

        # 检查是否有预定义的分割
        if isinstance(dataset_info['splits']['train'], (int, float)):
            # 如果没有预定义分割，我们默认使用 'train'
            self.split = 'train'
        else:
            # 如果有预定义分割，则尝试获取指定分割，如果不存在，则使用原始分割
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

    # def __getitem__(self, index):
    #     sample = self.hf_dataset[index]
    #
    #     # Extract image and text using the keys from the dictionary
    #     image = sample[self.keys['image']]
    #     text = sample[self.keys['text']]

    # def __getitem__(self, index):
    #     # image = Image.open(self.images[index])
    #     # caption = self.captions[index]
    #     sample = self.hf_dataset[index]
    #
    #     # # Extract image and text using the keys from the dictionary
    #     # image = sample[self.keys['image']]
    #     # Check if 'url' is in the key for image data to determine if it's a URL
    #     if 'url' in self.keys['image']:
    #         # Load image from URL
    #         print(sample[self.keys['image']])
    #         response = requests.get(sample[self.keys['image']])
    #         image = Image.open(BytesIO(response.content))
    #     else:
    #         # already a PIL Image object
    #         image = sample[self.keys['image']]
    #
    #
    #     caption = sample[self.keys['text']]
    #     if isinstance(caption, list):
    #         caption = random.choice(caption)
    #     caption = self.tokenize(caption)
    #     if self.transforms:
    #         image = self.transforms(image)
    #
    #     return {"image": image, "caption": caption}

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
            text = text[0]

        # text = self.tokenize(text)

        if self.transforms:
            image = self.transforms(image)

        return {"image": image, "text": text}
