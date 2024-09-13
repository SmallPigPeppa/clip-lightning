import os
import json
from .base import ImageRetrievalDataset


class COCO2014Dataset(ImageRetrievalDataset):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def fetch_dataset(self, split):
        self.root_dir = os.path.join(self.root_dir, 'coco')
        json_path = os.path.join(self.root_dir, 'annotations', 'dataset.json')
        with open(json_path, 'r') as file:
            all_data = json.load(file)['images']

        if split == 'train':
            split_data = [item for item in all_data if item['split'] in ['train', 'restval']]
        elif split == 'val':
            split_data = [item for item in all_data if item['split'] == 'test']
        else:
            raise ValueError('Split must be either "train" or "val"')

        images = []
        captions = []

        for item in split_data:
            img = os.path.join(self.root_dir, item['filepath'], item['filename'])
            caps = [sentence['raw'] for sentence in item['sentences']]
            assert os.path.isfile(img)
            images.append(img)
            captions.append(caps)

        return images, captions
