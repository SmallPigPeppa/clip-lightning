import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from model_openai import my_load
from zero_shot.zero_shot_classifier import ZeroShotClassifier
from model_openai import SimpleTokenizer
from zero_shot.zero_shot_metadata_imagenet import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from timm.utils import accuracy
from tqdm import tqdm
import random
from typing import Union, List


def recall_i2t_torch(image_feats: torch.Tensor, text_feats: torch.Tensor, caps_per_image: int) -> float:
    """
    图→文 R@1
    image_feats: (N, D)
    text_feats:  (N*c, D)
    """
    sims = image_feats @ text_feats.T  # 1. 计算相似度 (N, N*c)
    top1 = sims.argmax(dim=1)  # 2. 每图最匹配 caption 的索引 (N,)
    pred_img = top1 // caps_per_image  # 3. caption idx → 图 idx
    N = image_feats.size(0)
    correct = pred_img == torch.arange(N, device=image_feats.device)  # 4. 召回率 = 命中数 / N
    return correct.float().mean().item() * 100.0


def recall_t2i_torch(image_feats: torch.Tensor, text_feats: torch.Tensor, caps_per_image: int) -> float:
    """
    文→图 R@1
    image_feats: (N, D)
    text_feats:  (N*c, D)
    """
    sims = text_feats @ image_feats.T  # 1. 计算相似度 (N*c, N)
    top1 = sims.argmax(dim=1)  # 2. 每 caption 最匹配 图像 的索引 (N*c,)
    M = text_feats.size(0)
    true_img = torch.arange(M, device=text_feats.device) // caps_per_image  # 3. caption idx → 图 idx
    correct = top1 == true_img  # 4. 召回率 = 命中数 / (N*c)
    return correct.float().mean().item() * 100.0


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'RN50',
            download_root: str = None,
            projection_dims: int = 1024,
            temperature: float = 1.0,
            weight_decay: float = 0.0,
            lr_visual: float = 1e-3,
            lr_text: float = 5e-4,
            lr_warmup_epochs: int = 5,
            batch_size: int = 64,
            old_checkpoint_path: Union[str, List[str]] = None,
            current_task: int = 0,
            batch_size_zs: int = 256,
            zero_shot_eval_interval: int = 5,
            recall_eval_interval: int = 5,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = my_load(name=model_name, download_root=download_root)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.setup_msun(list(range(32, 96, 16)), list(range(32, 96, 16)), list(range(32, 96, 16)), 32)

    def setup_msun(self, res1_list, res2_list, res3_list, unified_size):
        """
        Embed MSUN into model.visual with:
          - 1 random‐subnet encoder: encode_image_randres
          - 3 fixed encoders:       encode_image_res1/2/3
        No input interpolation—subnets are applied directly.
        """
        visual = self.model.visual

        # build shared stem & tail
        stem = nn.Sequential(
            visual.conv1, visual.bn1, visual.relu1,
            visual.conv2, visual.bn2, visual.relu2,
            visual.conv3, visual.bn3, visual.relu3,
            visual.avgpool, visual.layer1
        )
        tail = nn.Sequential(visual.layer2, visual.layer3, visual.layer4, visual.attnpool)

        # attach subnets + configs
        visual.subnet1 = copy.deepcopy(stem)
        visual.subnet2 = copy.deepcopy(stem)
        visual.subnet3 = copy.deepcopy(stem)
        visual.unified_net = tail

        visual.res1_list = res1_list
        visual.res2_list = res2_list
        visual.res3_list = res3_list
        visual.unified_size = unified_size

        def encode_image_res1(self, x):
            z = self.subnet1(x)
            z_u = F.interpolate(z, size=self.unified_size, mode='bilinear', align_corners=False)
            y = self.unified_net(z_u)
            return z, y

        def encode_image_res2(self, x):
            z = self.subnet2(x)
            z_u = F.interpolate(z, size=self.unified_size, mode='bilinear', align_corners=False)
            y = self.unified_net(z_u)
            return z, y

        def encode_image_res3(self, x):
            z = self.subnet3(x)
            z_u = F.interpolate(z, size=self.unified_size, mode='bilinear', align_corners=False)
            y = self.unified_net(z_u)
            return z, y

        # bind to visual class
        for name, fn in [
            ('encode_image_res1', encode_image_res1),
            ('encode_image_res2', encode_image_res2),
            ('encode_image_res3', encode_image_res3),
        ]:
            setattr(visual.__class__, name, fn)
        delattr(visual.__class__, 'encode_image')


    def forward(self, inputs):
        images = inputs["image"]
        captions = inputs["caption"]
        # multi-caption
        if isinstance(captions, list):
            captions = random.choice(captions)
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(captions)

        return image_features, text_features

    def mixedres_forward(self, inputs):
        imgs, caps = inputs["image"], inputs["caption"]
        if isinstance(caps, list):
            caps = random.choice(caps)

        b, c, h, w = imgs.shape
        s = random.randint(32, 224)
        # 下采样到 (s, s)
        down = F.interpolate(imgs, size=(s, s), mode='bilinear', align_corners=False)
        # 恢复到原始尺寸 (h, w)
        up = F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False)

        img_feats = self.model.encode_image(up)
        txt_feats = self.model.encode_text(caps)
        return img_feats, txt_feats

    def configure_optimizers(self):
        lr_visual = self.hparams.lr_visual
        lr_text = self.hparams.lr_text
        min_lr = min(lr_visual, lr_text)

        parameters = [
            {
                "params": self.model.visual.parameters(),
                "lr": lr_visual
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "visual" not in n],
                "lr": lr_text,
                "weight_decay": self.hparams.weight_decay
            }
        ]

        optimizer = optim.AdamW(parameters, weight_decay=self.hparams.weight_decay)

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.lr_warmup_epochs,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=0.01 * min_lr,
            eta_min=0.01 * min_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def _compute_losses(self, image_features, text_features):
        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp().to(self.device)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]

        labels = torch.arange(len(logits_per_image)).to(self.device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        loss = (image_loss + text_loss) / 2

        return loss

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.mixedres_forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        # self.log("train/clip_loss", clip_loss, sync_dist=True)
        self.log("train/clip_loss", clip_loss)

        return clip_loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.mixedres_forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        # self.log("val/clip_loss", clip_loss, sync_dist=True)
        self.log("val/clip_loss", clip_loss)

        return clip_loss

    def on_train_start(self):
        # recall metric
        self.model.eval()
        val_loader = self.trainer.datamodule.val_dataloader()
        recall_metric = self.get_recall_metrics(val_loader)
        # self.log_dict(recall_metric, sync_dist=True)
        self.log_dict(recall_metric)

        # # Zero-shot metric evaluation before training starts
        # zero_shot_loader = self.trainer.datamodule.zero_shot_dataloader()
        # zero_shot_metric = self.get_zero_shot_metrics(zero_shot_loader)
        # self.log_dict(zero_shot_metric, sync_dist=True)
        self.model.train()

    def on_validation_epoch_end(self):
        # recall metric
        self.model.eval()
        if (self.current_epoch + 1) % self.hparams.recall_eval_interval == 0:
            val_loader = self.trainer.datamodule.val_dataloader()
            recall_metric = self.get_recall_metrics(val_loader)
            # self.log_dict(recall_metric, sync_dist=True)
            self.log_dict(recall_metric)

        # # zero-shot metric
        # if (self.current_epoch + 1) % self.hparams.zero_shot_eval_interval == 0:
        #     zero_shot_loader = self.trainer.datamodule.zero_shot_dataloader()
        #     zero_shot_metric = self.get_zero_shot_metrics(zero_shot_loader)
        #     self.log_dict(zero_shot_metric, sync_dist=True)
        self.model.train()

    def get_recall_metrics(self, dataloader):
        img_feats, txt_feats = [], []
        with torch.no_grad():
            for batch in dataloader:
                imgs = batch["image"].to(self.device)  # [B, C, H, W] → GPU
                # 展平所有 captions, [1,2,3][1,2,3] -> [112233]
                caps_tensor = torch.stack(batch["caption"], dim=0)
                # swap & flatten to [batch_size*caps_per_image, …]
                caps = caps_tensor.transpose(0, 1).flatten(0, 1).to(self.device)

                img_feats.append(self.model.encode_image(imgs))  # [B, D]
                txt_feats.append(self.model.encode_text(caps))  # [B*C, D]

        imgs = torch.cat(img_feats, dim=0)  # [N, D]
        txts = torch.cat(txt_feats, dim=0)  # [N*C, D]
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)  # 归一化
        txts = txts / txts.norm(dim=-1, keepdim=True)  # 归一化

        C = len(dataloader.dataset[0]["caption"])  # 每图 caption 数
        a = {
            "val/image_to_text_R@1": recall_i2t_torch(imgs, txts, C),  # 图→文
            "val/text_to_image_R@1": recall_t2i_torch(imgs, txts, C),  # 文→图
        }
        print(a)
        return {
            "val/image_to_text_R@1": recall_i2t_torch(imgs, txts, C),  # 图→文
            "val/text_to_image_R@1": recall_t2i_torch(imgs, txts, C),  # 文→图
        }

    def get_zero_shot_metrics(self, dataloader):
        self.tokenizer = SimpleTokenizer()
        self.zero_shot_classifier = ZeroShotClassifier(
            model=self.model,
            tokenizer=self.tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=self.hparams.batch_size_zs,
        ).to(self.device)

        self.zero_shot_classifier.compute_weights()

        top1, top5, n = 0., 0., 0.

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Zero-shot Evaluating", unit="batch"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.zero_shot_classifier(images)
                # Measure accuracy
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                top1 += acc1.item() * images.size(0)
                top5 += acc5.item() * images.size(0)
                n += images.size(0)

        top1 = top1 / n
        top5 = top5 / n
        metrics = {
            "zero_shot/top1_accuracy": top1,
            "zero_shot/top5_accuracy": top5
        }
        # Release the zero-shot classifier model to free up GPU memory
        del self.zero_shot_classifier
        return metrics


import copy
import torch.nn.functional as F
from torch import nn
