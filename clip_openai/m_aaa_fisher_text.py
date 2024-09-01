from torch.nn.functional import cross_entropy
import lightning as pl
import torch
from model_openai import my_load
from model_openai import SimpleTokenizer
from zero_shot.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from timm.utils import accuracy
from tqdm import tqdm
from packaging import version
from dataloaders.imagenet import build_dataset
from torch.utils.data import DataLoader
import argparse
import os
from lightning.pytorch.loggers import WandbLogger


class YourLightningModule(pl.LightningModule):

    def __init__(
            self,
            model_name: str = 'RN50',
            download_root: str = None,
            num_classes_per_batch: int = 10,
            max_length: int = 77,
            lr: float = 1e-9,
            batch_size: int = 32,
            num_workers: int = 8,
            root_dir: str = './data',
            fisher_dir: str = './fisher_data',
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.model = my_load(
            name=model_name,
            download_root=download_root
        )
        self.tokenizer = SimpleTokenizer()
        self.classnames = IMAGENET_CLASSNAMES
        self.templates = OPENAI_IMAGENET_TEMPLATES
        self.num_classes_per_batch = num_classes_per_batch
        self.max_length = max_length
        self.zeroshot_weights = None
        self.fisher = None

        for param in self.model.visual.parameters():
            param.requires_grad = False

        # # fix text model
        # for param in self.model.transformer.parameters():
        #     param.requires_grad = False
        #
        # for param in self.model.token_embedding.parameters():
        #     param.requires_grad = False
        #
        # # 冻结 positional_embedding 参数
        # self.model.positional_embedding.requires_grad = False
        #
        # # 冻结 ln_final 参数
        # for param in self.model.ln_final.parameters():
        #     param.requires_grad = False

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

        # with torch.no_grad():
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

    def forward(self, images):
        self.compute_weights()
        if self.zeroshot_weights is None:
            raise ValueError("Zero-shot weights not computed. Call `compute_weights` first.")
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        logits = 100. * image_features @ self.zeroshot_weights
        return logits

    def on_train_epoch_start(self):
        self.compute_weights()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        images, targets = batch
        # Compute logits and loss
        logits = self.forward(images)
        loss = cross_entropy(logits, targets)

        # Compute gradients
        self.manual_backward(loss)

        # Initialize or update Fisher information matrix
        if self.fisher is None:
            self.fisher = {
                n: torch.zeros_like(p).to(self.device) for n, p in self.model.named_parameters()
                if p.requires_grad
            }

        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] += p.grad.pow(2)

        opt.step()

        # Log training loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        metrics = {
            "train/top1_accuracy": acc1,
            "train/top5_accuracy": acc5,
            "train/loss": loss
        }
        self.log_dict(metrics, sync_dist=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # Normalize Fisher Information by the number of batches
        for n in self.fisher:
            self.fisher[n] /= len(self.train_dataloader())
            # Calculate the mean of the Fisher information across all dimensions
            self.fisher[n] = self.fisher[n].mean()

        print(self.fisher)
        # Optionally, save fisher information after each epoch
        os.makedirs(self.hparams.fisher_dir, exist_ok=True)
        model_name_safe = self.hparams.model_name.replace("/", "_")
        fisher_save_path = os.path.join(self.hparams.fisher_dir,
                                        f"{model_name_safe}_text_fisher_epoch_{self.current_epoch}.pth")

        # Save the Fisher information
        torch.save(self.fisher, fisher_save_path)
        print(f"Fisher information saved to {fisher_save_path}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        # Define your data loader here
        args = argparse.Namespace(
            data_set='IMNET',  # Specify ImageNet dataset
            data_path=os.path.join(self.hparams.root_dir, 'imagenet'),  # Path to your ImageNet dataset
            input_size=224,  # Example of image size, adjust according to your needs
            eval_crop_ratio=0.875  # Example of crop percentage, adjust according to your needs
            # Add other relevant parameters as needed
        )
        self.zero_shot_dataset, nb_classes = build_dataset(
            is_train=False,
            args=args
        )

        return DataLoader(
            self.zero_shot_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )


if __name__ == '__main__':
    model = YourLightningModule(
        model_name='ViT-B/16',
        download_root='./',
        num_classes_per_batch=20,
        lr=0.,
        max_length=77,
        batch_size=128,
        num_workers=8,
        root_dir='/ppio_net0/torch_ds',
        fisher_dir='./fisher_data',

    )
    wandb_logger = WandbLogger(
        name='fisher',
        project='fisher',
        offline=False,
        log_model=False
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        num_nodes=1,
        log_every_n_steps=1,
        precision=16,
        max_epochs=1,
        logger=wandb_logger,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        strategy='ddp_find_unused_parameters_true'

    )
    trainer.fit(model)
