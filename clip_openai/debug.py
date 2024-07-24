from dataloaders import Flickr30kDataset
from clip_openai.model_openai.simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    from clip_openai.model_openai.clip_old import my_load
    from dataloaders import ImageRetrievalDataModule

    dm = ImageRetrievalDataModule(
        dataset_name='flickr30k',
        config='../config.yaml',
        root_dir='../artifacts/flickr-30k:v0',
        val_split=0.2,
        max_length=77
    )
    dm.setup(stage='fit')
    a = dm.train_dataloader()
    inputs = next(iter(a))

    model = my_load(name='RN50', download_root='./')
    image_features = model.encode_image(inputs["image"])
    text_features = model.encode_text(inputs["caption"])
    print(image_features.shape)
    print(text_features.shape)