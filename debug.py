from image_retrieval.dataloaders.data_module_aug import ImageRetrievalDataModule as ImageRetrievalDataModule_AUG
from image_retrieval.dataloaders.data_module import ImageRetrievalDataModule

if __name__ == "__main__":
    artifact_id = ''
    dataset_name = 'flickr30k_aug'
    config = './config.yaml'
    tokenizer_alias = 'distilbert-base-uncased'
    dm = ImageRetrievalDataModule_AUG(artifact_id=artifact_id,
                                  dataset_name=dataset_name,
                                  config=config,
                                  tokenizer_alias=tokenizer_alias)
    dm.setup()
    a = dm.train_dataloader()
    print(next(iter(a)))
    b = dm.val_dataloader()
    c = b.dataset
    # print(b.dataset.dataset.images)


    artifact_id = ''
    dataset_name = 'flickr30k'
    config = './config.yaml'
    tokenizer_alias = 'distilbert-base-uncased'
    dm2 = ImageRetrievalDataModule(artifact_id=artifact_id,
                                  dataset_name=dataset_name,
                                  tokenizer_alias=tokenizer_alias)
    dm2.setup()
    a2 = dm2.train_dataloader()
    print(next(iter(a2)))
    b2 = dm2.val_dataloader()
    c2 = b2.dataset
    print(b2.dataset.dataset.images)
