from abc import abstractmethod
import torch
from PIL import Image
from torch.utils.data import Dataset
from packaging import version
from typing import Union, List


class ImageRetrievalDataset(Dataset):
    def __init__(
            self, root_dir, tokenizer, max_length: int = 200, transforms=None
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.images, self.captions = self.fetch_dataset()

    def __len__(self):
        return len(self.captions)

    # def tokenize(self, text):
    #     sot_token = self.tokenizer.encoder["<|startoftext|>"]
    #     eot_token = self.tokenizer.encoder["<|endoftext|>"]
    #     tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
    #     result = torch.zeros(self.max_length, dtype=torch.int)
    #     if len(tokens) <= self.max_length:
    #         result[:len(tokens)] = torch.tensor(tokens)
    #     else:
    #         result[:self.max_length] = torch.tensor(tokens)[:self.max_length]
    #     return result

    def tokenize(self, texts: Union[str, List[str]], truncate: bool = False) -> Union[
        torch.IntTensor, torch.LongTensor]:
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        if version.parse(torch.__version__) < version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), self.max_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), self.max_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_length:
                if truncate:
                    tokens = tokens[:self.max_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {self.max_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        caption = self.tokenize(self.captions[index])
        if self.transforms:
            image = self.transforms(image)

        return {"image": image, "caption": caption}

    @abstractmethod
    def fetch_dataset(self):
        pass
