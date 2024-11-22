import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from model_openai import my_load
from zero_shot.zero_shot_classifier_l2p import ZeroShotClassifier
from model_openai import SimpleTokenizer
from zero_shot.zero_shot_metadata_imagenet import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from timm.utils import accuracy
from tqdm import tqdm
import copy
import math


# 定义 L2P 的 Prompt 模块
class PromptModule(nn.Module):
    def __init__(self, prompt_length, embed_dim):
        super(PromptModule, self).__init__()
        # self.prompt_embeddings = nn.Parameter(torch.zeros(prompt_length, embed_dim))
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, embed_dim))

    def forward(self, x):
        batch_size = x.size(0)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([prompt, x], dim=1)
        return x


def resize_pos_embed(pos_embed, new_num_tokens, num_prefix_tokens=1):
    """
    Resize positional embeddings with bicubic interpolation.

    Args:
        pos_embed (torch.Tensor): 原始位置嵌入 (N, D)。
        new_num_tokens (int): 新的 token 数量。
        num_prefix_tokens (int): 前缀 token 的数量（如 CLS token）。

    Returns:
        torch.Tensor: 调整后的位置嵌入 (N_new, D)。
    """
    # 拆分前缀嵌入和网格嵌入
    pos_prefix = pos_embed[:num_prefix_tokens, :]  # 提取前缀 token 的嵌入
    pos_grid = pos_embed[num_prefix_tokens:, :]  # 提取网格部分的嵌入

    # 计算原始网格大小（假设为正方形）
    num_grid_tokens = pos_grid.size(0)
    grid_size_old = int(math.sqrt(num_grid_tokens))
    # grid_size_new = int(math.sqrt(new_num_tokens - num_prefix_tokens))
    # 向上取整计算新的网格大小
    grid_size_new = math.ceil(math.sqrt(new_num_tokens - num_prefix_tokens))

    # 调整形状以适配插值 (C, H, W)
    pos_grid = pos_grid.reshape(grid_size_old, grid_size_old, -1).permute(2, 0, 1)

    # 使用插值调整网格大小
    pos_grid = F.interpolate(pos_grid.unsqueeze(0), size=(grid_size_new, grid_size_new), mode='bicubic',
                             align_corners=False)

    # 恢复到原始形状 (N_new, D)
    pos_grid = pos_grid.squeeze(0).permute(1, 2, 0).reshape(grid_size_new ** 2, -1)

    # 合并前缀和网格嵌入
    pos_embed_new = torch.cat([pos_prefix, pos_grid], dim=0)

    return pos_embed_new


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
            lr_project: float = 1e-5,
            lr_warmup_epochs: int = 5,
            batch_size: int = 64,
            old_checkpoint_path: str = None,
            current_task: int = 0,
            batch_size_zs: int = 256,
            zero_shot_eval_interval: int = 5,
            recall_eval_interval: int = 5,
            prompt_length: int = 5,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = my_load(name=model_name, download_root=download_root)
        # self.initialize_old_modules()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        # 初始化文本和视觉部分的 Prompt 模块
        embed_dim_text = self.model.transformer.width
        embed_dim_visual = self.model.visual.transformer.width

        self.prompt_module_text = PromptModule(self.hparams.prompt_length, embed_dim_text)
        self.prompt_module_visual = PromptModule(self.hparams.prompt_length, embed_dim_visual)

        # 冻结原始模型参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 使 Prompt 模块的参数可训练
        for param in self.prompt_module_text.parameters():
            param.requires_grad = True

        for param in self.prompt_module_visual.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        # 编码图像
        image = inputs["image"]
        image_features = self.encode_image_with_prompt(image)

        # 编码文本
        text = inputs["caption"]
        text_features = self.encode_text_with_prompt(text)

        return image_features, text_features

    def update_attn_mask(self, new_length):
        """
        根据新序列长度动态调整所有 ResidualAttentionBlock 的 attn_mask。
        :param new_length: 新的序列长度（添加 Prompt 后的长度）。
        """
        for block in self.model.transformer.resblocks:
            # 动态生成新的 attn_mask
            new_attn_mask = torch.triu(torch.ones(new_length, new_length), diagonal=1).to(
                block.attn_mask.device) * float("-inf")
            block.attn_mask = new_attn_mask

    '''
    def encode_text(self, text):
    x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

    x = x + self.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    '''

    def encode_text_with_prompt(self, text_tokens):
        """
        编码带有 Prompt 的文本序列。
        :param text_tokens: 输入的文本 tokens，形状为 [batch_size, n_ctx]
        :return: 文本特征，形状为 [batch_size, d_model]
        """

        text_features = self.model.encode_text(text_tokens)

        return text_features
        # 获取词嵌入
        x = self.model.token_embedding(text_tokens)  # [batch_size, n_ctx, d_model]

        # 添加 Prompt
        x = self.prompt_module_text(x)  # 假设 prompt_module_text 会在序列前添加 prompt

        # 原始序列长度和添加 Prompt 后的长度
        original_length = text_tokens.size(1)
        extended_length = x.size(1)  # 添加 Prompt 后的序列长度

        self.update_attn_mask(extended_length)

        # 插值位置编码
        original_pos_embed = self.model.positional_embedding[:original_length, :]  # [n_ctx, d_model]
        # 使用线性插值扩展位置编码
        interpolated_pos_embed = F.interpolate(
            original_pos_embed.unsqueeze(0).permute(0, 2, 1),  # 转换为 [1, d_model, n_ctx]
            size=extended_length,  # 插值到 extended_length
            mode='linear',
            align_corners=False
        ).squeeze(0).permute(1, 0).to(x.device)  # [extended_length, d_model]

        # 添加位置编码
        pos_embed = interpolated_pos_embed.unsqueeze(0)  # [1, extended_length, d_model]
        x = x + pos_embed

        x = x.permute(1, 0, 2)  # NLD -> LND

        # Transformer
        x = self.model.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        # 计算 eot_token 的位置
        # 原始 eot_token 的位置是 text_tokens.argmax(dim=-1)
        # 加上 Prompt 的长度偏移 self.hparams.prompt_length
        eot_positions = text_tokens.argmax(dim=-1) + self.hparams.prompt_length

        # 提取 eot_token 的特征
        x = x[torch.arange(x.size(0)), eot_positions]  # [batch_size, d_model]

        # 归一化
        x = self.model.ln_final(x)

        # 线性投影到特征空间
        text_features = x @ self.model.text_projection

        return text_features

    '''
        def forward(self, x: torch.Tensor):
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=self.device),
                 x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding
            x = self.ln_pre(x)
    
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
    
            x = self.ln_post(x[:, 0, :])
    
            if self.proj is not None:
                x = x @ self.proj
    
            return x
    '''

    def encode_image_with_prompt(self, image):
        # image_features = self.model.encode_image(image)
        # return image_features
        x = self.model.visual.conv1(image)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # 添加 Prompt
        x = self.prompt_module_visual(x)

        # 添加类嵌入
        class_embedding = self.model.visual.class_embedding.to(x.dtype)
        class_embedding = class_embedding.unsqueeze(0).unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([class_embedding, x], dim=1)

        # print('self.model.visual.positional_embedding',self.model.visual.positional_embedding.shape)
        # 调整位置嵌入以适配新 token 数
        pos_embed = resize_pos_embed(
            self.model.visual.positional_embedding,
            new_num_tokens=x.size(1),
            num_prefix_tokens=1
        ).to(x.device, x.dtype)

        # print('x', x.shape)
        # print('pos_embed', pos_embed.shape)
        pos_embed = pos_embed[:x.size(1), :].unsqueeze(0).to(x.device)
        x = x + pos_embed

        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.model.visual.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.model.visual.ln_post(x[:, 0, :])

        if self.model.visual.proj is not None:
            x = x @ self.model.visual.proj

        return x

    def configure_optimizers(self):
        # 只优化 Prompt 模块的参数
        parameters = [
            {
                "params": self.prompt_module_visual.parameters(),
                "lr": self.hparams.lr
            },
            {
                "params": self.prompt_module_text.parameters(),
                "lr": self.hparams.lr_text,
                "weight_decay": self.hparams.weight_decay
            }
        ]
        # parameters = [{
        #     "params": self.model.parameters(),
        #     "lr": self.hparams.lr,
        #     "weight_decay": self.hparams.weight_decay
        # }]
        optimizer = optim.AdamW(parameters, weight_decay=self.hparams.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.lr_warmup_epochs,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=1 * self.hparams.lr,
            eta_min=1 * self.hparams.lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def _compute_losses(self, image_features, text_features):

        # 归一化特征
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算余弦相似度作为 logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]

        labels = torch.arange(len(logits_per_image)).to(self.device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        loss = (image_loss + text_loss) / 2

        return loss

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        self.log("train/clip_loss", clip_loss, sync_dist=True)

        return clip_loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        self.log("val/clip_loss", clip_loss, sync_dist=True)

        return clip_loss

    def on_train_start(self):
        # 计算 Recall 指标
        val_loader = self.trainer.datamodule.val_dataloader()
        recall_metric = self.get_recall_metrics(val_loader)
        self.log_dict(recall_metric, sync_dist=True)

    def on_validation_epoch_end(self):
        # 计算 Recall 指标
        if (self.current_epoch + 1) % self.hparams.recall_eval_interval == 0:
            val_loader = self.trainer.datamodule.val_dataloader()
            recall_metric = self.get_recall_metrics(val_loader)
            self.log_dict(recall_metric, sync_dist=True)

        # 计算 Zero-shot 指标
        if (self.current_epoch + 1) % self.hparams.zero_shot_eval_interval == 0:
            zero_shot_loader = self.trainer.datamodule.zero_shot_dataloader()
            zero_shot_metric = self.get_zero_shot_metrics(zero_shot_loader)
            self.log_dict(zero_shot_metric, sync_dist=True)

    def get_recall_metrics(self, dataloader):
        val_img_feats = []
        val_text_feats = []

        with torch.no_grad():
            for inputs in tqdm(dataloader, desc="Recall Evaluating", unit="batch"):
                images = inputs["image"].to(self.device)
                targets = inputs["caption"].to(self.device)
                image_features = self.encode_image_with_prompt(images)
                text_features = self.encode_text_with_prompt(targets)
                val_img_feats.append(image_features)
                val_text_feats.append(text_features)

        all_image_features = torch.cat(val_img_feats)
        all_text_features = torch.cat(val_text_feats)

        metrics = self.recall_score(
            image_features=all_image_features,
            text_features=all_text_features,
            logit_scale=self.model.logit_scale.exp(),
        )

        return metrics

    def recall_score(self, image_features, text_features, logit_scale=1.0):
        metrics = {}
        logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
        logits_per_text = logits_per_image.t().detach().cpu()

        logits = {"val/image_to_text": logits_per_image, "val/text_to_image": logits_per_text}
        ground_truth = torch.arange(len(text_features)).view(-1, 1)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == ground_truth)[1]
            preds = preds.detach().cpu().numpy()
            for k in [1]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k) * 100  # 将 Recall 转换为百分比

        return metrics

    def get_zero_shot_metrics(self, dataloader):
        self.tokenizer = SimpleTokenizer()
        self.zero_shot_classifier = ZeroShotClassifier(
            model=self.model,
            prompt_module_text=self.prompt_module_text,
            prompt_module_visual=self.prompt_module_visual,
            tokenizer=self.tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=self.hparams.batch_size_zs,
            prompt_length=self.hparams.prompt_length
        ).to(self.device)

        self.zero_shot_classifier.compute_weights()

        top1, top5, n = 0., 0., 0.

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Zero-shot Evaluating", unit="batch"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.zero_shot_classifier(images)
                # 计算准确率
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
        # 释放内存
        del self.zero_shot_classifier
        return metrics

    # def on_save_checkpoint(self, checkpoint):
    #     if self.trainer.current_epoch != self.trainer.max_epochs - 1:
    #         pass
    #     elif self.trainer.current_epoch == self.trainer.max_epochs - 1:
    #         # 保存 Prompt 模块的参数
    #         if self.trainer.is_global_zero:
    #             print('************************')
    #
    #             # 创建一个新的 state_dict 用于保存权重
    #             new_state_dict = {}
    #
    #             # 保存模型的参数
    #             for name, param in self.model.named_parameters():
    #                 new_state_dict[name] = param.data
    #
    #             # 保存 Prompt 模块的参数
    #             for name, param in self.prompt_module_text.named_parameters():
    #                 new_state_dict[f"prompt_module_text.{name}"] = param.data
    #
    #             for name, param in self.prompt_module_visual.named_parameters():
    #                 new_state_dict[f"prompt_module_visual.{name}"] = param.data
    #
    #             # 将处理后的权重保存到检查点
    #             checkpoint['model'] = new_state_dict
    #
    #             print('Saved model parameters with prompt modules:')
    #             for name in new_state_dict.keys():
    #                 print(name)
    #
    #             print('************************')
    #
    #             # 比较新模型参数名与旧模型参数名
    #             print('Parameter Comparison with Old Model:')
    #             for (new_name, param), (old_name, old_param) in zip(new_state_dict.items(),
    #                                                                 self.model_old.named_parameters()):
    #                 if new_name != old_name:
    #                     print(f"New: {new_name} | Old: {old_name}")
    #
    #             print('************************')
