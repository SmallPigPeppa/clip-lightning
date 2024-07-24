import os
import pandas as pd
from .base import ImageRetrievalDataset


class Flickr30kDataset(ImageRetrievalDataset):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

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


if __name__ == "__main__":
    from clip_openai.model import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    dataset = Flickr30kDataset(
        root_dir='../../artifacts/flickr-30k:v0',
        tokenizer=tokenizer,
    )
    a = dataset[0]
    print(a)
