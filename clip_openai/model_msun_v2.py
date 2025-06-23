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
        self.setup_msun(
            res_lists=[
                list(range(32, 96, 16)),
                list(range(96, 160, 16)),
                list(range(160, 224, 16)),
                [224],
            ],
            unified_size=56,
        )

    def setup_msun(self, res_lists: List[List[int]], unified_size: int):
        visual = self.model.visual

        # split backbone
        stem = nn.Sequential(
            visual.conv1, visual.bn1, visual.relu1,
            visual.conv2, visual.bn2, visual.relu2,
            visual.conv3, visual.bn3, visual.relu3,
            visual.avgpool, visual.layer1
        )
        tail = nn.Sequential(visual.layer2, visual.layer3, visual.layer4, visual.attnpool)

        # attach unified tail
        visual.unified_net = tail
        visual.unified_size = unified_size
        self.num_subnets = len(res_lists)

        # create subnets and encoders
        for i, res in enumerate(res_lists, 1):
            subnet = copy.deepcopy(stem)
            # inline customization for first three subnets
            if i in (1, 2, 3):
                subnet[0].stride = (1, 1)
            if i == 1:
                subnet[9] = nn.Identity()

            setattr(visual, f"subnet{i}", subnet)
            setattr(visual, f"res{i}_list", res)

            # bind fixed-res encoder
            def make_encoder(j):
                def encode(self, x):
                    z = getattr(self.visual, f"subnet{j}")(x)
                    y = self.visual.unified_net(
                        F.interpolate(z, self.visual.unified_size,
                                      mode='bilinear', align_corners=False)
                    )
                    return z, y

                return encode

            setattr(self.model.__class__, f"encode_image_res{i}", make_encoder(i))

        # original-resolution encoder via last subnet
        def encode_image(self, x):
            return self.visual.unified_net(
                getattr(self.visual, f"subnet{self.num_subnets}")(x)
            )

        setattr(self.model.__class__, 'encode_image', encode_image)

    def forward(self, inputs):
        imgs, caps = inputs['image'], inputs['caption']
        if isinstance(caps, list):
            caps = random.choice(caps)

        _, _, h, _ = imgs.shape
        for idx in range(1, self.num_subnets + 1):
            if h in getattr(self.model.visual, f'res{idx}_list'):
                _, img_feats = getattr(self.model, f'encode_image_res{idx}')(imgs)
                break

        txt_feats = self.model.encode_text(caps)
        return img_feats, txt_feats

    def randres_forward(self, inputs):
        imgs, caps = inputs['image'], inputs['caption']
        if isinstance(caps, list):
            caps = random.choice(caps)

        z_list, feat_list = [], []
        # 对每个 subnet 的列表都采样一个分辨率并路由
        for idx in range(1, self.num_subnets + 1):
            res_list = getattr(self.model.visual, f'res{idx}_list')
            r = random.choice(res_list)
            down = F.interpolate(imgs, size=(r, r), mode='bilinear', align_corners=False)
            z_i, img_i = getattr(self.model, f'encode_image_res{idx}')(down)
            z_list.append(z_i)
            feat_list.append(img_i)

        txt_feats = self.model.encode_text(caps)
        return z_list, feat_list, txt_feats

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

    def _compute_losses_sir(self, z1, z2):
        z1_u = F.interpolate(z1, size=self.model.visual.unified_size, mode='bilinear', align_corners=False)
        z2_u = F.interpolate(z2, size=self.model.visual.unified_size, mode='bilinear', align_corners=False)
        # print(z1.shape, z2.shape)
        loss = self.mse_loss(z1_u, z2_u)
        return loss

    def training_step(self, batch, *args, **kwargs):
        # get per-subnet features and texts
        z_list, feat_list, txt = self.randres_forward(batch)

        # compute clip loss for each subnet
        clip_losses = [self._compute_losses(f, txt) for f in feat_list]
        # compute sir loss against last subnet as reference
        ref = z_list[-1]
        sir_losses = [self._compute_losses_sir(z, ref) for z in z_list[:-1]]

        # aggregate totals
        total_clip = sum(clip_losses)
        total_sir = sum(sir_losses)

        # log individual subnet losses
        for i, l in enumerate(clip_losses, 1):
            self.log(f"train/clip_loss_subnet{i}", l)
        for i, l in enumerate(sir_losses, 1):
            self.log(f"train/sir_loss_subnet{i}", l)

        # log aggregated losses
        self.log("train/clip_loss", total_clip)
        self.log("train/sir_loss", total_sir)

        # return weighted sum
        return total_clip + self.hparams.alpha * total_sir

    def validation_step(self, batch, *args, **kwargs):
        # get per-subnet features and texts
        z_list, feat_list, txt = self.randres_forward(batch)

        # compute clip loss per subnet
        clip_losses = [self._compute_losses(f, txt) for f in feat_list]
        # compute sir loss against last subnet as reference
        ref = z_list[-1]
        sir_losses = [self._compute_losses_sir(z, ref) for z in z_list[:-1]]

        # aggregate totals
        total_clip = sum(clip_losses)
        total_sir = sum(sir_losses)

        # log individual subnet losses
        for i, l in enumerate(clip_losses, 1):
            self.log(f"val/clip_loss_subnet{i}", l)
        for i, l in enumerate(sir_losses, 1):
            self.log(f"val/sir_loss_subnet{i}", l)

        # log aggregated losses
        self.log("val/clip_loss", total_clip)
        self.log("val/sir_loss", total_sir)

        # return weighted sum
        return total_clip + self.hparams.alpha * total_sir

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
                    imgs_r = F.interpolate(imgs, size=(r, r), mode='bilinear', align_corners=False)

                    # route through matching subnet
                    for i in range(1, self.num_subnets + 1):
                        if r in getattr(self.model.visual, f"res{i}_list"):
                            _, feats = getattr(self.model, f"encode_image_res{i}")(imgs_r)
                            break
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

