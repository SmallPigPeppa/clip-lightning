# from dataloaders import Flickr30kDataset
# from clip_openai.model_openai.simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    from dataloaders.data_module_dil_json import ImageRetrievalDataModule
    dm = ImageRetrievalDataModule(
        dataset_name='flickr30k',
        config='../config.yaml',
        root_dir='/Users/lwz/torch_ds',
        max_length=77,
        batch_size=4
    )
    dm.setup(stage='validate')
    #
    # b = dm.task_datasets[0]
    # print(len(b))
    # c = dm.val_dataset
    # print(len(c))
    # a = dm.train_dataloader()
    # inputs = next(iter(a))
    # print(len(inputs['caption']))

    a = dm.val_dataloader()
    inputs = next(iter(a))

    print(len(inputs['caption']))
    print(inputs['caption'][0].shape)


