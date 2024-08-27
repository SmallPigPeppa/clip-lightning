import torch
from torch import nn
from lightning import LightningModule
from typing import Sequence, Callable, Union, Optional


class ZeroShotClassifier(LightningModule):
    def __init__(
            self,
            model,
            tokenizer,
            classnames: Sequence[str],
            templates: Sequence[Union[Callable, str]],
            num_classes_per_batch: Optional[int] = 10,
            use_tqdm: bool = True,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.templates = templates
        self.num_classes_per_batch = num_classes_per_batch
        self.use_tqdm = use_tqdm
        # self.zeroshot_weights = None
        self.compute_weights()

    def forward(self, images):
        if self.zeroshot_weights is None:
            raise ValueError("Zero-shot weights not computed. Call `compute_weights` first.")
        image_features = self.model.encode_image(images)
        logits = 100. * image_features @ self.zeroshot_weights
        return logits

    def compute_weights(self):
        use_format = isinstance(self.templates[0], str)
        num_templates = len(self.templates)
        num_classes = len(self.classnames)

        def _process_batch(batch_classnames):
            texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in
                     self.templates]
            texts = self.tokenizer(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings.reshape(len(batch_classnames), num_templates, -1).mean(dim=1)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
            class_embeddings = class_embeddings.T
            return class_embeddings

        with torch.no_grad():
            if self.num_classes_per_batch:
                batched_embeds = [_process_batch(batch) for batch in
                                  self._batch_classes(self.classnames, self.num_classes_per_batch)]
                self.zeroshot_weights = torch.cat(batched_embeds, dim=1)
            else:
                self.zeroshot_weights = _process_batch(self.classnames)

    def _batch_classes(self, classnames, num_classes_per_batch):
        return [classnames[i:i + num_classes_per_batch] for i in range(0, len(classnames), num_classes_per_batch)]
