from dataloaders import Flickr30kDataset
from clip_openai.model.simple_tokenizer import SimpleTokenizer


if __name__ == "__main__":
    tokenizer = SimpleTokenizer()
    dataset = Flickr30kDataset(
        root_dir='../artifacts/flickr-30k:v0',
        tokenizer=tokenizer,
    )
    a = dataset[0]
    print(a)
