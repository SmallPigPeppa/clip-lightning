from dataloaders import Flickr30kDataset
from clip_openai.model_openai.simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    from clip_openai.model_openai.clip_old import my_load
    model = my_load(name='ViT-B/16', download_root='./')

    # 输出模型中所有参数的名称
    for name, param in model.named_parameters():
        print(name)

