import os
import pandas as pd
from .base import ImageRetrievalDataset


class Flickr30kDataset(ImageRetrievalDataset):
    def __init__(
            self,
            root_dir: str = None,
            tokenizer=None,
            max_length: int = 100,
            transforms=None,
    ) -> None:
        super().__init__(root_dir, tokenizer, max_length, transforms)

    def fetch_dataset(self):
        annotations = pd.read_csv(os.path.join(self.root_dir, "results.csv"), sep='|')
        annotations = annotations.dropna()
        image_files = [
            os.path.join(self.root_dir, "flickr30k_images", image_file)
            for image_file in annotations["image_name"].to_list()
        ]
        for image_file in image_files:
            assert os.path.isfile(image_file)
        captions = annotations[" comment"].tolist()
        return image_files, captions
