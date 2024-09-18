import os
import pandas as pd
from .base_av import ImageRetrievalDataset
import json


class Flickr30kDataset(ImageRetrievalDataset):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)


    def fetch_dataset(self, split):
        self.root_dir = os.path.join(self.root_dir, 'flickr30k')
        json_path = os.path.join(self.root_dir, 'dataset.json')
        with open(json_path, 'r') as file:
            all_data = json.load(file)['images']

        if split == 'train':
            split_data = [item for item in all_data if item['split'] in ['train', 'val']]
        elif split == 'val':
            split_data = [item for item in all_data if item['split'] == 'test']
        else:
            raise ValueError('Split must be either "train" or "val"')

        images = []
        captions = []

        for item in split_data:
            img = os.path.join(self.root_dir, "flickr30k_images", item['filename'])
            caps = [sentence['raw'] for sentence in item['sentences']]
            assert os.path.isfile(img)
            images.append(img)
            captions.append(caps)

        return images, captions


if __name__ == "__main__":
    from clip_openai.model import SimpleTokenizer

    tokenizer = SimpleTokenizer()
    dataset = Flickr30kDataset(
        root_dir='../../artifacts/flickr-30k:v0/flickr30k_images',
        tokenizer=tokenizer,
    )
    a = dataset[0]
    print(a)
