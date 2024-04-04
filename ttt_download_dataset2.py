# import os
import wandb
# import pandas as pd
# from typing import Optional
#
# from image_retrieval.dataloaders.base import ImageRetrievalDataset


# class Flickr30kDataset(ImageRetrievalDataset):
#     def __init__(
#             self,
#             artifact_id: str,
#             tokenizer=None,
#             target_size: Optional[int] = None,
#             max_length: int = 100,
#             lazy_loading: bool = False,
#     ) -> None:
#         super().__init__(artifact_id, tokenizer, target_size, max_length, lazy_loading)
#
#     def fetch_dataset(self):
#         if wandb.run is None:
#             api = wandb.Api()
#             artifact = api.artifact(self.artifact_id, type="dataset")
#         else:
#             artifact = wandb.use_artifact(self.artifact_id, type="dataset")
#         artifact_dir = artifact.download()
#         annotations = pd.read_csv(os.path.join(artifact_dir, "results.csv"), sep='|')
#         annotations = annotations.dropna()
#         image_files = [
#             os.path.join(artifact_dir, "flickr30k_images", image_file)
#             for image_file in annotations["image_name"].to_list()
#         ]
#         for image_file in image_files:
#             assert os.path.isfile(image_file)
#         captions = annotations[" comment"].tolist()
#         return image_files, captions


if __name__ == '__main__':
    artifact_id = 'wandb/clip.lightning-image_retrieval/flickr-30k:latest'
    api = wandb.Api()
    artifact = api.artifact(artifact_id, type="dataset")
    # wandb.init()
    # artifact = wandb.use_artifact(artifact_id, type="dataset")
    artifact_dir = artifact.download()
