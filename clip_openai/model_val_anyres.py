import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from model_openai import my_load
from typing import Union, List
import wandb
from lightning.pytorch.loggers import WandbLogger


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
        self.val_img_feats = []
        self.val_text_feats = []

    def validation_step(self, batch, *args, **kwargs):
        clip_loss = 0.
        # self.log("val/clip_loss", clip_loss, sync_dist=True)
        self.log("val/clip_loss", clip_loss)

        return clip_loss

    def log_recall_table(self, recall_list):
        """
        recall_list: [
            {"resolution": 32, "image2text_recall": 45.0, "text2image_recall": 50.0},
            {"resolution": 48, "image2text_recall": 47.2, "text2image_recall": 52.1},
            ...
        ]
        """
        # 列名：第一列叫 "Metric"，后面每列用分辨率标识
        columns = ["Metric"] + [str(d["resolution"]) for d in recall_list]
        table = wandb.Table(columns=columns)

        # 要记录的 recall 指标
        metrics_list = ["image2text_recall", "text2image_recall"]

        # 每个指标做一行
        for name in metrics_list:
            row = [name]
            for d in recall_list:
                # 默认不存在时填 0
                row.append(d.get(name, 0.0))
            table.add_data(*row)

        import pdb;pdb.set_trace()
        wandb.log({"recall": table})

    def on_validation_epoch_end(self):
        val_loader = self.trainer.datamodule.val_dataloader()
        resolutions = list(range(32, 225, 16))

        # 收集各分辨率的 recall
        recall_list = []
        for s in resolutions:
            m = self.get_recall_metrics_anyres(val_loader, s)
            recall_list.append({
                "resolution": s,
                "image2text_recall": m["val/image_to_text_R@1"],
                "text2image_recall": m["val/text_to_image_R@1"],
            })

        # 调用上面的函数将表格 log 到 W&B
        self.log_recall_table(recall_list)

    def get_recall_metrics_anyres(self, dataloader, s=None):
        img_feats, txt_feats = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Recall @ {s}px", unit="batch"):
                imgs = batch["image"].to(self.device)  # [B, C, H, W] → GPU
                if s is not None:
                    b, c, h, w = imgs.shape
                    down = F.interpolate(imgs, size=(s, s), mode='bilinear', align_corners=False)
                    imgs = F.interpolate(down, size=(h, w), mode='bilinear', align_corners=False)

                caps_tensor = torch.stack(batch["caption"], dim=0)
                caps = caps_tensor.transpose(0, 1).flatten(0, 1).to(self.device)

                img_feats.append(self.model.encode_image(imgs))  # [B, D]
                txt_feats.append(self.model.encode_text(caps))  # [B*C, D]

        imgs = torch.cat(img_feats, dim=0)  # [N, D]
        txts = torch.cat(txt_feats, dim=0)  # [N*C, D]
        imgs = imgs / imgs.norm(dim=-1, keepdim=True)  # 归一化
        txts = txts / txts.norm(dim=-1, keepdim=True)  # 归一化

        C = len(dataloader.dataset[0]["caption"])  # 每图 caption 数
        # C = 1 for debug
        # txts = txts.repeat_interleave(C, dim=0)
        a = {
            "val/image_to_text_R@1": recall_i2t_torch(imgs, txts, C),  # 图→文
            "val/text_to_image_R@1": recall_t2i_torch(imgs, txts, C),  # 文→图
        }
        print(a)
        return {
            "val/image_to_text_R@1": recall_i2t_torch(imgs, txts, C),  # 图→文
            "val/text_to_image_R@1": recall_t2i_torch(imgs, txts, C),  # 文→图
        }
