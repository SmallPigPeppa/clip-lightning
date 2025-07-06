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
import copy


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
            alpha: float = 10,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = my_load(name=model_name, download_root=download_root)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.mse_loss = nn.MSELoss()
        self.setup_mixedres(
            res_lists=[
                list(range(32, 81, 16)),
                list(range(96, 145, 16)),
                list(range(160, 209, 16)),
                [224],
            ],
        )

    def setup_mixedres(self, res_lists: List[List[int]]):
        visual = self.model.visual
        self.num_subnets = len(res_lists)
        # create res_list
        for i, res in enumerate(res_lists, 1):
            setattr(visual, f"res{i}_list", res)

    def forward(self, inputs):
        images = inputs["image"]
        captions = inputs["caption"]
        # multi-caption
        if isinstance(captions, list):
            captions = random.choice(captions)
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(captions)

        return image_features, text_features

    def randres_forward(self, inputs):
        imgs, caps = inputs['image'], inputs['caption']

        if isinstance(caps, list):
            caps = random.choice(caps)

        feat_list = []
        b, c, h, w = imgs.shape
        # 对每个 subnet 的列表都采样一个分辨率并路由
        for idx in range(1, self.num_subnets + 1):
            res_list = getattr(self.model.visual, f'res{idx}_list')
            r = random.choice(res_list)
            down = F.interpolate(imgs, size=(r, r), mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False)
            img_i = self.model.encode_image(up)
            feat_list.append(img_i)

        txt_feats = self.model.encode_text(caps)
        return feat_list, txt_feats

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
        # forward pass for each resolution
        feat_list, txt = self.randres_forward(batch)

        # compute CLIP losses for all subnets
        clip_losses = [self._compute_losses(f, txt) for f in feat_list]

        # aggregate losses
        total_clip = sum(clip_losses)

        # log individual losses
        for i, l in enumerate(clip_losses, 1):
            self.log(f"train/clip_loss_subnet{i}", l)

        # log totals
        self.log("train/clip_loss", total_clip)

        # return combined loss
        return total_clip

    def validation_step(self, batch, *args, **kwargs):
        # forward pass for each resolution
        feat_list, txt = self.randres_forward(batch)

        # compute CLIP losses for all subnets
        clip_losses = [self._compute_losses(f, txt) for f in feat_list]

        # aggregate losses
        total_clip = sum(clip_losses)

        # log individual losses
        for i, l in enumerate(clip_losses, 1):
            self.log(f"val/clip_loss_subnet{i}", l)

        # log totals
        self.log("val/clip_loss", total_clip)

        # return combined loss
        return total_clip

    def on_train_start(self):
        self.model.eval()
        val_loader = self.trainer.datamodule.val_dataloader()
        recall_metric = self.get_recall_metrics(val_loader)
        self.log_dict(recall_metric)
        self.model.train()

    def on_validation_epoch_end(self):
        self.model.eval()
        if (self.current_epoch + 1) % self.hparams.recall_eval_interval == 0:
            val_loader = self.trainer.datamodule.val_dataloader()
            recall_metric = self.get_recall_metrics(val_loader)
            self.log_dict(recall_metric)
        self.model.train()

    def get_recall_metrics(self, dataloader):
        """
        Compute R@1 for image-to-text and text-to-image at preset resolutions.
        """
        res_sizes = [32, 128, 176, 224]
        results = {}

        for r in res_sizes:
            img_feats, txt_feats = [], []
            with torch.no_grad():
                for batch in dataloader:
                    imgs = batch["image"].to(self.device)
                    b, c, h, w = imgs.shape
                    down = F.interpolate(imgs, size=(r, r), mode='bilinear', align_corners=False)
                    up = F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False)
                    feats = self.model.encode_image(up)

                    img_feats.append(feats)

                    caps = torch.stack(batch["caption"], dim=0)
                    caps = caps.transpose(0, 1).flatten(0, 1).to(self.device)
                    txt_feats.append(self.model.encode_text(caps))

            imgs_cat = torch.cat(img_feats, dim=0)
            txts_cat = torch.cat(txt_feats, dim=0)
            imgs_norm = imgs_cat / imgs_cat.norm(dim=-1, keepdim=True)
            txts_norm = txts_cat / txts_cat.norm(dim=-1, keepdim=True)
            caps_per_image = len(dataloader.dataset[0]["caption"])

            i2t = recall_i2t_torch(imgs_norm, txts_norm, caps_per_image)
            t2i = recall_t2i_torch(imgs_norm, txts_norm, caps_per_image)

            results[f"val/R{r}_i2t_R@1"] = i2t
            results[f"val/R{r}_t2i_R@1"] = t2i

        return results
