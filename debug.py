from image_retrieval.dataloaders.data_module_aug import ImageRetrievalDataModule

if __name__ == "__main__":
    artifact_id = ''
    dataset_name = 'flickr30k_aug'
    config = './config.yaml'
    tokenizer_alias = 'distilbert-base-uncased'
    dm = ImageRetrievalDataModule(artifact_id=artifact_id,
                                  dataset_name=dataset_name,
                                  config=config,
                                  tokenizer_alias=tokenizer_alias)
    dm.setup()
    a = dm.train_dataloader()
    print(next(iter(a)))