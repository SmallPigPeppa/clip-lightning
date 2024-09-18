# from dataloaders import Flickr30kDataset
# from clip_openai.model_openai.simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    from dataloaders.data_module_dil_json import ImageRetrievalDataModule

    # Initialize the DataModule with relevant parameters
    dm = ImageRetrievalDataModule(
        dataset_name='coco2014',
        config='./config.yaml',
        root_dir='/ppio_net0/torch_ds',
        max_length=77,
        batch_size=128
    )

    # Set up the DataModule for the validation stage
    dm.setup(stage='validate')

    # Access the validation dataset directly
    val_dataset = dm.val_dataset

    # List to store the lengths of all captions in the validation set
    caption_lengths = []

    # Iterate through the validation dataset
    for i in range(len(val_dataset)):
        data_item = val_dataset[i]  # Get the data item at index i
        caption = data_item['caption']  # Get the caption for the data item
        if isinstance(caption, list):
            caption_lengths.append(len(caption))  # Store the length of each caption if it's a list
        else:
            caption_lengths.append(1)  # Store the length of the caption if it's a single caption

    # Print out all the caption lengths
    print(f"Total captions: {len(caption_lengths)}")
    print("Caption lengths:", caption_lengths)
