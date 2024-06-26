import os
import wandb
import pandas as pd
from typing import Optional

from .base_aug import ImageRetrievalDataset


class Flickr30kDataseAug(ImageRetrievalDataset):
    def __init__(
            self,
            artifact_id: str,
            tokenizer=None,
            max_length: int = 100,
            transforms=None
    ) -> None:
        super().__init__(artifact_id, tokenizer, max_length, transforms)

    def fetch_dataset(self):
        # if wandb.run is None:
        #     api = wandb.Api()
        #     artifact = api.artifact(self.artifact_id, type="dataset")
        # else:
        #     artifact = wandb.use_artifact(self.artifact_id, type="dataset")
        # artifact_dir = artifact.download()
        artifact_dir = 'artifacts/flickr-30k:v0'
        annotations = pd.read_csv(os.path.join(artifact_dir, "results.csv"), sep='|')
        annotations = annotations.dropna()
        image_files = [
            os.path.join(artifact_dir, "flickr30k_images", image_file)
            for image_file in annotations["image_name"].to_list()
        ]
        for image_file in image_files:
            assert os.path.isfile(image_file)
        captions = annotations[" comment"].tolist()
        return image_files, captions
