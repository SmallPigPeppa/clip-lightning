import torch
import transformers
from torch import nn
# from transformers import DistilBertTokenizer, DistilBertModel
from .modeling_distillbert_plv2 import DistilBertModel_PL as DistilBertModel
import lightning as pl

# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertModel.from_pretrained("distilbert-base-uncased")


# class TextEncoder(nn.Module):
class TextEncoder(pl.LightningModule):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()

        # self.model = transformers.AutoModel.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]



