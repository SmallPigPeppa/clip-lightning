import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from model_openai import my_load
import random
from typing import Union, List
import copy
import numpy as np
from types import MethodType


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
        self.setup_msun(list(range(32, 96, 1)), list(range(96, 159, 1)), list(range(160, 225, 1)), 56)
        # self.setup_s4pool()

    def setup_msun(self, res1_list, res2_list, res3_list, unified_size):
        # grab backbone
        visual = self.model.visual

        # shared stem & tail
        stem = nn.Sequential(
            visual.conv1, visual.bn1, visual.relu1,
            visual.conv2, visual.bn2, visual.relu2,
            visual.conv3, visual.bn3, visual.relu3,
            visual.avgpool, visual.layer1
        )
        tail = nn.Sequential(visual.layer2, visual.layer3, visual.layer4, visual.attnpool)

        # clone stems into subnets
        visual.subnet1 = copy.deepcopy(stem)
        visual.subnet2 = copy.deepcopy(stem)
        visual.subnet3 = copy.deepcopy(stem)
        visual.unified_net = tail

        # make subnet1/2 conv1 stride =1
        visual.subnet1[0].stride = (1, 1)
        visual.subnet1[9] = nn.Identity()
        visual.subnet2[0].stride = (1, 1)
        visual.subnet3[0].stride = (1, 1)

        # resolution configs
        visual.res1_list = res1_list
        visual.res2_list = res2_list
        visual.res3_list = res3_list
        visual.unified_size = unified_size

        # fixed‐resolution encoders
        def encode_image_res1(self, x):
            z = self.visual.subnet1(x)
            y = self.visual.unified_net(
                F.interpolate(z, self.visual.unified_size, mode='bilinear', align_corners=False))
            return z, y

        def encode_image_res2(self, x):
            z = self.visual.subnet2(x)
            y = self.visual.unified_net(
                F.interpolate(z, self.visual.unified_size, mode='bilinear', align_corners=False))
            return z, y

        def encode_image_res3(self, x):
            z = self.visual.subnet3(x)
            import pdb;
            pdb.set_trace()
            y = self.visual.unified_net(
                F.interpolate(z, self.visual.unified_size, mode='bilinear', align_corners=False))
            return z, y

        # original‐resolution encoder
        def encode_image(self, x):
            return self.visual.unified_net(self.visual.subnet3(x))

        # bind to visual class
        for name, fn in [
            ('encode_image_res1', encode_image_res1),
            ('encode_image_res2', encode_image_res2),
            ('encode_image_res3', encode_image_res3),
            ('encode_image', encode_image),
        ]:
            setattr(self.model.__class__, name, fn)

    def setup_s4pool(self):
        def conv_s4pool(self, input):
            (shift1, shift2) = (np.random.randint(2), np.random.randint(2))
            if input.shape[2] % 2 == 1:
                padded_input = F.pad(input, (0, 0, self.padding[0], self.padding[0]), "constant", 0)
                shift1 = 0
            else:
                first_side = self.padding[0] - shift1
                second_side = self.padding[0] + shift1 - 1
                padded_input = F.pad(input, (0, 0, first_side, second_side), "constant", 0)
            if input.shape[3] % 2 == 1:
                shift2 = 0
                padded_input = F.pad(padded_input, (self.padding[1], self.padding[1], 0, 0), "constant", 0)
            else:
                first_side = self.padding[1] - shift2
                second_side = self.padding[1] + shift2 - 1
                padded_input = F.pad(padded_input, (first_side, second_side, 0, 0), "constant", 0)
            fmap = F.conv2d(
                padded_input,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups
            )
            return fmap

        def s4_maxpool(self, input):
            (shift1, shift2) = (np.random.randint(2), np.random.randint(2))
            if input.shape[2] % 2 == 1:
                padded_input = F.pad(input, (0, 0, self.padding, self.padding), "constant", float('-inf'))
            else:
                first_side = self.padding - shift1
                second_side = self.padding + shift1 - 1
                padded_input = F.pad(input, (0, 0, first_side, second_side), "constant", float('-inf'))
            if input.shape[3] % 2 == 1:
                padded_input = F.pad(padded_input, (self.padding, self.padding, 0, 0), "constant", float('-inf'))
            else:
                first_side = self.padding - shift2
                second_side = self.padding + shift2 - 1
                padded_input = F.pad(padded_input, (first_side, second_side, 0, 0), "constant", float('-inf'))
            return F.max_pool2d(
                padded_input, self.kernel_size, self.stride,
                0, self.dilation, self.ceil_mode,
                self.return_indices)

        def conv_s4pool_1x1(self, input):
            (shift1, shift2) = (np.random.randint(2), np.random.randint(2))
            fmap = F.conv2d(
                input[:, :, shift1:, shift2:],
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups
            )
            return fmap

        def modify_conv_module(mod):
            if isinstance(mod, torch.nn.Conv2d) and mod.kernel_size[0] > 1 and mod.stride == (2, 2):
                mod.forward = MethodType(conv_s4pool, mod)
            if isinstance(mod, torch.nn.Conv2d) and mod.kernel_size[0] == 1 and mod.stride == (2, 2):
                mod.forward = MethodType(conv_s4pool_1x1, mod)
            if isinstance(mod, torch.nn.MaxPool2d):
                print("changed maxpool")
                mod.forward = MethodType(s4_maxpool, mod)

        self.model.visual.apply(modify_conv_module)

    def forward(self, inputs):
        """
        Choose encode_image_res* based on input size and forward.
        """
        imgs, caps = inputs['image'], inputs['caption']
        if isinstance(caps, list):
            caps = random.choice(caps)

        _, _, h, w = imgs.shape
        # select subnet by resolution (lists now on self.model.visual)
        if h in self.model.visual.res1_list:
            _, img_feats = self.model.encode_image_res1(imgs)
        elif h in self.model.visual.res2_list:
            _, img_feats = self.model.encode_image_res2(imgs)
        else:
            _, img_feats = self.model.encode_image_res3(imgs)

        txt_feats = self.model.encode_text(caps)
        return img_feats, txt_feats

    def randres_forward(self, inputs):
        """
        Randomly pick one resolution from visual.res*_list,
        resize image, and forward through corresponding encode_image_res* subnet.
        """
        imgs, caps = inputs['image'], inputs['caption']
        if isinstance(caps, list):
            caps = random.choice(caps)
        import pdb; pdb.set_trace()
        z_3, img_feats = self.model.encode_image_res3(imgs)

        # sample one resolution from visual
        all_res = (
                self.model.visual.res1_list +
                self.model.visual.res2_list +
                self.model.visual.res3_list
        )
        r = random.choice(all_res)

        # down & up sample
        down = F.interpolate(imgs, size=(r, r), mode='bilinear', align_corners=False)

        # route to correct subnet
        if r in self.model.visual.res1_list:
            z_i, img_feats_i = self.model.encode_image_res1(down)
        elif r in self.model.visual.res2_list:
            z_i, img_feats_i = self.model.encode_image_res2(down)
        else:
            z_i, img_feats_i = self.model.encode_image_res3(down)

        txt_feats = self.model.encode_text(caps)

        return z_i, z_3, img_feats_i, img_feats, txt_feats

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
        z_i, z_3, img_feats_i, img_feats, txt_feats = self.randres_forward(batch)
        clip_loss = self._compute_losses(img_feats, txt_feats)
        sir_loss = self._compute_losses_sir(z_i, z_3)
        clip_loss_i = self._compute_losses(img_feats_i, txt_feats)
        self.log("train/clip_loss", clip_loss)
        self.log("train/sir_loss", sir_loss)
        self.log("train/clip_loss_i", clip_loss_i)

        return clip_loss + clip_loss_i + self.hparams.alpha * sir_loss

    def validation_step(self, batch, *args, **kwargs):
        z_i, z_3, img_feats_i, img_feats, txt_feats = self.randres_forward(batch)
        clip_loss = self._compute_losses(img_feats, txt_feats)
        sir_loss = self._compute_losses_sir(z_i, z_3)
        clip_loss_i = self._compute_losses(img_feats_i, txt_feats)
        self.log("val/clip_loss", clip_loss)
        self.log("val/sir_loss", sir_loss)
        self.log("val/clip_loss_i", clip_loss_i)

        return clip_loss + clip_loss_i + self.hparams.alpha * sir_loss

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
