import torch
from lightning import LightningModule
from typing import Sequence, Callable, Union, Optional
from packaging import version
from tqdm import tqdm

class ZeroShotClassifier(LightningModule):
    def __init__(
            self,
            model,
            tokenizer,
            classnames: Sequence[str],
            templates: Sequence[Union[Callable, str]],
            num_classes_per_batch: Optional[int] = 10,
            max_length: int = 77,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.templates = templates
        self.num_classes_per_batch = num_classes_per_batch
        self.max_length = max_length
        self.zeroshot_weights = None
        # self.compute_weights()

    def tokenize(self, text):
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        if version.parse(torch.__version__) < version.parse("1.8.0"):
            result = torch.zeros(self.max_length, dtype=torch.long)
        else:
            result = torch.zeros(self.max_length, dtype=torch.int)

        if len(tokens) <= self.max_length:
            result[:len(tokens)] = torch.tensor(tokens)
        else:
            result[:self.max_length] = torch.tensor(tokens)[:self.max_length]
        return result

    def forward(self, images):
        if self.zeroshot_weights is None:
            raise ValueError("Zero-shot weights not computed. Call `compute_weights` first.")
        image_features = self.model.encode_image(images)
        logits = 100. * image_features @ self.zeroshot_weights
        return logits

    from tqdm import tqdm

    def compute_weights(self):
        use_format = isinstance(self.templates[0], str)
        num_templates = len(self.templates)
        num_classes = len(self.classnames)

        def _process_batch(batch_classnames):
            texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in
                     self.templates]
            # texts = self.tokenizer.encode(texts).to(self.device)
            # print(self.device)
            texts = [self.tokenize(t).to(self.device) for t in texts]
            texts = torch.stack(texts)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings.reshape(len(batch_classnames), num_templates, -1).mean(dim=1)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
            class_embeddings = class_embeddings.T
            return class_embeddings

        with torch.no_grad():
            if self.num_classes_per_batch:
                batched_embeds = []
                for batch in tqdm(self.batch_classes(self.classnames, self.num_classes_per_batch),
                                  desc="Computing Zero-shot Weights", unit="batch"):
                    batched_embeds.append(_process_batch(batch))
                self.zeroshot_weights = torch.cat(batched_embeds, dim=1)
            else:
                self.zeroshot_weights = _process_batch(self.classnames)

    def batch_classes(self, classnames, num_classes_per_batch):
        return [classnames[i:i + num_classes_per_batch] for i in range(0, len(classnames), num_classes_per_batch)]
