from dataloaders import Flickr30kDataset

if __name__ == "__main__":
    dataset = Flickr30kDataset(
        root_dir='../artifacts/flickr-30k:v0')
    a = dataset[0]
