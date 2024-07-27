

if __name__ == "__main__":
    # from clip_openai.model_openai.clip_old import my_load
    from dataloaders.data_module_dil import ImageRetrievalDataModule

    dm = ImageRetrievalDataModule(
        dataset_name='coco2014',
        config='../config.yaml',
        root_dir='/Users/lwz/torch_ds/coco_caption',
        tokenizer_alias='distilbert-base-uncased',
        max_length=77
    )
    # dm = ImageRetrievalDataModule(
    #     dataset_name='flickr30k',
    #     config='../config.yaml',
    #     root_dir='../artifacts/flickr-30k:v0',
    #     val_split=0.2,
    #     max_length=77
    # )
    dm.setup(stage='fit')

    b = dm.task_datasets[0]
    print(len(b))
    c = dm.val_dataset
    print(len(c))
    # d = dm.train_dataset
    # print(len(d))

    a = dm.train_dataloader()
    inputs = next(iter(a))
    print(inputs)

    # model = my_load(name='RN50', download_root='./')
    # image_features = model.encode_image(inputs["image"])
    # text_features = model.encode_text(inputs["caption"])
    # print(image_features.shape)
    # print(text_features.shape)
