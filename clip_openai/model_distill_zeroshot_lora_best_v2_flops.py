import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from model_openai import my_load
import copy

from peft import get_peft_model, LoraConfig


def cognition_align(old, new):
    n, m = old.shape
    index = torch.max(old, 1)[1]  ### 找到对比矩阵中每一行的最大值索引
    for i in range(n):
        if index[i] != i:  ### 如果最大值索引不是当前行的索引值，则说明过去模型对当前该样本数据认知错误，为此将这一行替换为当前训练模型的对应矩阵行，排除其对当前训练模型的影响。
            old[i] = new[i]

    return old, new


def find_target_modules(model):
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            target_modules.append(name)
    return target_modules


def get_lora_model_vision(model):
    # Define the target modules where LoRA should be applied
    target_modules = find_target_modules(model)
    lora_config = LoraConfig(
        inference_mode=False,
        r=16,  # Rank of the low-rank decomposition
        lora_alpha=32,  # Scaling factor
        task_type='vision',  # Task type
        lora_dropout=0.1,  # Dropout rate for LoRA
        target_modules=target_modules,
    )
    # Apply LoRA to the model
    lora_model = get_peft_model(model, lora_config)

    return lora_model


def get_lora_model_text(model):
    # Define the target modules where LoRA should be applied
    target_modules = find_target_modules(model)

    # Initialize LoRA configuration with target modules
    lora_config = LoraConfig(
        inference_mode=False,
        r=16,  # Rank of the low-rank decomposition
        lora_alpha=32,  # Scaling factor
        task_type='text',  # Task type
        lora_dropout=0.1,  # Dropout rate for LoRA
        target_modules=target_modules  # Specify the target modules
    )

    # Apply LoRA to the model
    lora_model = get_peft_model(model, lora_config)

    return lora_model


class DistillPredictor(nn.Module):
    def __init__(self, projection_dims, distill_proj_hidden_dim):
        super(DistillPredictor, self).__init__()
        self.pd_linear1 = nn.Linear(projection_dims, distill_proj_hidden_dim)
        self.pd_batch_norm = nn.BatchNorm1d(distill_proj_hidden_dim)
        self.pd_relu = nn.ReLU()
        self.pd_linear2 = nn.Linear(distill_proj_hidden_dim, projection_dims)

    def forward(self, x):
        x = self.pd_linear1(x)
        x = self.pd_batch_norm(x)
        x = self.pd_relu(x)
        x = self.pd_linear2(x)
        return x


class NewModel(nn.Module):
    def __init__(self, original_conv1):
        super(NewModel, self).__init__()
        # 直接使用 deep copy 复制 conv1 层
        self.conv1 = copy.deepcopy(original_conv1)

    def forward(self, x):
        x = self.conv1(x)
        return x


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'RN50',
            download_root: str = None,
            projection_dims: int = 1024,
            temperature: float = 1.0,
            weight_decay: float = 0.0,
            lr: float = 1e-3,
            lr_text: float = 5e-4,
            lr_warmup_epochs: int = 5,
            batch_size: int = 64,
            old_checkpoint_path: str = None,
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
        self.distill = True
        self.initialize_old_modules()
        self.model.transformer = get_lora_model_text(self.model.transformer)
        self.model.visual.transformer = get_lora_model_vision(self.model.visual.transformer)

    def initialize_old_modules(self):
        if self.hparams.old_checkpoint_path is not None:
            if ',' in self.hparams.old_checkpoint_path:
                self.hparams.old_checkpoint_path = self.hparams.old_checkpoint_path.split(',')
                avg_params = None
                count = 0
                for chkpt_path in self.hparams.old_checkpoint_path:
                    checkpoint = torch.load(chkpt_path, map_location=torch.device('cpu'))
                    model_params = checkpoint['model']

                    if avg_params is None:
                        avg_params = {k: v.clone().detach() for k, v in model_params.items()}
                    else:
                        for k in avg_params.keys():
                            avg_params[k] += model_params[k]

                    count += 1
                for k in avg_params.keys():
                    avg_params[k] /= count
                self.model.load_state_dict(avg_params, strict=True)
                print("Model weights loaded successfully and old parts averaged.")
            else:
                checkpoint = torch.load(self.hparams.old_checkpoint_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(checkpoint['model'], strict=True)
                print("Model weights loaded successfully and old parts copied.")

        self.model_old = copy.deepcopy(self.model)
        # Set requires_grad to False for all parameters in the old modules
        for param in self.model_old.parameters():
            param.requires_grad = False

        distill_proj_hidden_dim = 2048
        self.distill_predictor = DistillPredictor(
            projection_dims=self.hparams.projection_dims,
            distill_proj_hidden_dim=distill_proj_hidden_dim
        )
        if not self.distill:
            for param in self.distill_predictor.parameters():
                param.requires_grad = False


    def forward(self, inputs):
        image_features = self.model.encode_image(inputs["image"])
        text_features = self.model.encode_text(inputs["caption"])
        return image_features, text_features

    def forward_old(self, inputs):
        with torch.no_grad():
            image_features = self.model_old.encode_image(inputs["image"])
            text_features = self.model_old.encode_text(inputs["caption"])
        return image_features, text_features

    def configure_optimizers(self):
        parameters = [
            {
                "params": self.model.visual.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": [param for name, param in self.model.named_parameters() if "visual" not in name],
                "lr": self.hparams.lr_text,
                "weight_decay": self.hparams.weight_decay
            }
        ]

        if self.distill:
            parameters.append({
                "params": self.distill_predictor.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay
            })
        else:
            for param in self.distill_predictor.parameters():
                param.requires_grad = False

        optimizer = optim.AdamW(parameters, weight_decay=self.hparams.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.lr_warmup_epochs,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=0.01 * self.hparams.lr,
            eta_min=0.01 * self.hparams.lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def _compute_losses(self, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]

        labels = torch.arange(len(logits_per_image)).to(self.device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        loss = (image_loss + text_loss) / 2

        return loss

    def ckc_loss_func(
            self,
            p1: torch.Tensor,
            p2: torch.Tensor,
            z1: torch.Tensor,
            z2: torch.Tensor,
            # temperature: float = 0.1,
    ) -> torch.Tensor:

        device = self.device
        logit_scale = self.model.logit_scale.exp()

        b = z1.size(0)

        p = F.normalize(torch.cat([p1, p2]), dim=-1)
        z = F.normalize(torch.cat([z1, z2]), dim=-1)

        logits = torch.einsum("if, jf -> ij", p, z) * logit_scale
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
        pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
        pos_mask.fill_diagonal_(True)

        # all matches excluding the main diagonal
        logit_mask = torch.ones_like(pos_mask, device=device)
        logit_mask.fill_diagonal_(True)
        logit_mask[:, b:].fill_diagonal_(True)
        logit_mask[b:, :].fill_diagonal_(True)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        # loss
        loss = -mean_log_prob_pos.mean()
        return loss

    # def training_step(self, batch, *args, **kwargs):
    #     image_embeddings, text_embeddings = self.forward(batch)
    #     clip_loss = self._compute_losses(image_embeddings, text_embeddings)
    #     self.log("train/clip_loss", clip_loss, sync_dist=True)
    #
    #     if self.distill:
    #         frozen_z1, frozen_z2 = self.forward_old(batch)
    #         p1 = self.distill_predictor(image_embeddings)
    #         p2 = self.distill_predictor(text_embeddings)
    #
    #         distill_loss = (
    #                                self.ckc_loss_func(p1, p2, frozen_z1, frozen_z2)
    #                                + self.ckc_loss_func(frozen_z1, frozen_z2, p1, p2)
    #                        ) / 2
    #
    #         self.log("train/distill_loss", distill_loss, sync_dist=True)
    #         return clip_loss + distill_loss
    #     else:
    #         return clip_loss
    #
    # def validation_step(self, batch, *args, **kwargs):
    #     image_embeddings, text_embeddings = self.forward(batch)
    #     clip_loss = self._compute_losses(image_embeddings, text_embeddings)
    #     self.log("val/clip_loss", clip_loss, sync_dist=True)
    #
    #     if self.distill:
    #         frozen_z1, frozen_z2 = self.forward_old(batch)
    #         p1 = self.distill_predictor(image_embeddings)
    #         p2 = self.distill_predictor(text_embeddings)
    #
    #         distill_loss = (
    #                                self.ckc_loss_func(p1, p2, frozen_z1, frozen_z2)
    #                                + self.ckc_loss_func(frozen_z1, frozen_z2, p1, p2)
    #                        ) / 2
    #         self.log("val/distill_loss", distill_loss, sync_dist=True)
    #         return clip_loss + distill_loss
    #     else:
    #         return clip_loss


    # def zscl_distill(self, image_features, text_features, image_features_old, text_features_old, temperature=1.0,
    #                  alpha=0.5):
    #     # 归一化图像和文本特征
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=1, keepdim=True)
    #
    #     # 计算新模型的 logits（图像和文本）
    #     logit_scale = self.model.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logits_per_image.t()
    #
    #     # 归一化旧模型的图像和文本特征
    #     image_features_old = image_features_old / image_features_old.norm(dim=1, keepdim=True)
    #     text_features_old = text_features_old / text_features_old.norm(dim=1, keepdim=True)
    #
    #     # 计算旧模型的 logits（图像和文本）
    #     logit_scale_old = self.model.logit_scale.exp()
    #     logits_per_image_old = logit_scale_old * image_features_old @ text_features_old.t()
    #     logits_per_text_old = logits_per_image_old.t()
    #
    #     # 蒸馏损失（使用 KL 散度）——针对图像 logits 和文本 logits 分别计算
    #     # 使用软化的 softmax 作为输入，温度参数可以控制 logits 的平滑度
    #     distill_loss_image = F.kl_div(
    #         F.log_softmax(logits_per_image / temperature, dim=1),
    #         F.softmax(logits_per_image_old / temperature, dim=1),
    #         reduction='batchmean'
    #     ) * (temperature ** 2)  # KL 散度标准化
    #
    #     distill_loss_text = F.kl_div(
    #         F.log_softmax(logits_per_text / temperature, dim=1),
    #         F.softmax(logits_per_text_old / temperature, dim=1),
    #         reduction='batchmean'
    #     ) * (temperature ** 2)  # KL 散度标准化
    #
    #     # 总蒸馏损失可以是图像和文本部分的加权和
    #     distill_loss = alpha * distill_loss_image + (1 - alpha) * distill_loss_text
    #
    #     return distill_loss
    #
    # def training_step(self, batch, *args, **kwargs):
    #     image_embeddings, text_embeddings = self.forward(batch)
    #     clip_loss = self._compute_losses(image_embeddings, text_embeddings)
    #     self.log("train/clip_loss", clip_loss, sync_dist=True)
    #
    #     if self.distill:
    #         frozen_z1, frozen_z2 = self.forward_old(batch)
    #         distill_loss = self.zscl_distill(
    #             image_features=image_embeddings,
    #             text_features=text_embeddings,
    #             image_features_old=frozen_z1,
    #             text_features_old=frozen_z2
    #         )
    #
    #         self.log("train/distill_loss", distill_loss, sync_dist=True)
    #         return clip_loss + distill_loss * 2.0
    #     else:
    #         return clip_loss
    #
    # def validation_step(self, batch, *args, **kwargs):
    #     image_embeddings, text_embeddings = self.forward(batch)
    #     clip_loss = self._compute_losses(image_embeddings, text_embeddings)
    #     self.log("val/clip_loss", clip_loss, sync_dist=True)
    #
    #     if self.distill:
    #         frozen_z1, frozen_z2 = self.forward_old(batch)
    #         distill_loss = self.zscl_distill(
    #             image_features=image_embeddings,
    #             text_features=text_embeddings,
    #             image_features_old=frozen_z1,
    #             text_features_old=frozen_z2
    #         )
    #         self.log("val/distill_loss", distill_loss, sync_dist=True)
    #         return clip_loss + distill_loss * 2.0
    #     else:
    #         return clip_loss





    def modx_distill(self, image_features, text_features, image_features_old, text_features_old):
        ### 构造新旧模型在当前样本上的对比矩阵
        per_image = image_features @ text_features.t()
        per_text = text_features @ image_features.t()

        img_text_old = image_features_old @ text_features_old.t()
        text_img_old = text_features_old @ image_features_old.t()

        ### 套用 cognition_align 模块剔除错误认知信息
        logits_img_text_old, logits_per_image = cognition_align(img_text_old, per_image)
        logits_text_img_old, logits_per_text = cognition_align(text_img_old, per_text)

        off_dia_per_image = logits_per_image
        off_dia_img_text_old = logits_img_text_old

        off_dia_per_text = logits_per_text
        off_dia_text_img_old = logits_text_img_old

        ### 使用KL散度函数对非对角线信息分布进行对齐
        loss_dia_img = F.kl_div(off_dia_per_image.softmax(dim=-1).log(), off_dia_img_text_old.softmax(dim=-1),
                                reduction='sum')
        loss_dia_text = F.kl_div(off_dia_per_text.softmax(dim=-1).log(), off_dia_text_img_old.softmax(dim=-1),
                                 reduction='sum')

        loss_distill = (loss_dia_img + loss_dia_text) / 2
        return loss_distill

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        self.log("train/clip_loss", clip_loss, sync_dist=True)

        if self.distill:
            frozen_z1, frozen_z2 = self.forward_old(batch)
            p1 = self.distill_predictor(image_embeddings)
            p2 = self.distill_predictor(text_embeddings)

            # distill_loss = (
            #                        self.simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2)
            #                        + self.simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2)
            #                ) / 2
            distill_loss = self.modx_distill(
                image_features=image_embeddings,
                text_features=text_embeddings,
                image_features_old=frozen_z1,
                text_features_old=frozen_z2
            )

            self.log("train/distill_loss", distill_loss, sync_dist=True)
            return clip_loss + distill_loss * 0.005
        else:
            return clip_loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        self.log("val/clip_loss", clip_loss, sync_dist=True)

        if self.distill:
            frozen_z1, frozen_z2 = self.forward_old(batch)
            p1 = self.distill_predictor(image_embeddings)
            p2 = self.distill_predictor(text_embeddings)

            # distill_loss = (
            #                        self.simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2)
            #                        + self.simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2)
            #                ) / 2
            distill_loss = self.modx_distill(
                image_features=image_embeddings,
                text_features=text_embeddings,
                image_features_old=frozen_z1,
                text_features_old=frozen_z2
            )
            self.log("val/distill_loss", distill_loss, sync_dist=True)
            return clip_loss + distill_loss * 0.005
        else:
            return clip_loss







