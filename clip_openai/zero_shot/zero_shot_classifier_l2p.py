import torch
from lightning import LightningModule
from typing import Sequence, Callable, Union, Optional
from tqdm import tqdm
import copy
import torch.nn.functional as F
import math


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


class ZeroShotClassifier(LightningModule):
    def __init__(
            self,
            model,
            tokenizer,
            classnames: Sequence[str],
            templates: Sequence[Union[Callable, str]],
            prompt_module_text: Optional[torch.nn.Module] = None,
            prompt_module_visual: Optional[torch.nn.Module] = None,
            num_classes_per_batch: Optional[int] = 10,
            max_length: int = 77,
            prompt_length: int = 5

    ):
        super().__init__()
        self.model = copy.deepcopy(model).eval()
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.templates = templates
        self.prompt_module_text = prompt_module_text
        self.prompt_module_visual = prompt_module_visual
        self.num_classes_per_batch = num_classes_per_batch
        self.max_length = max_length
        self.zeroshot_weights = None
        self.prompt_length = prompt_length

    def tokenize(self, text):
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.max_length, dtype=torch.int)
        if len(tokens) <= self.max_length:
            result[:len(tokens)] = torch.tensor(tokens)
        else:
            result[:self.max_length] = torch.tensor(tokens)[:self.max_length]
        return result

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
        eot_positions = text_tokens.argmax(dim=-1) + self.prompt_length

        # 提取 eot_token 的特征
        x = x[torch.arange(x.size(0)), eot_positions]  # [batch_size, d_model]

        # 归一化
        x = self.model.ln_final(x)

        # 线性投影到特征空间
        text_features = x @ self.model.text_projection

        return text_features

    def encode_image_with_prompt(self, image):
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
        #
        # x = self.model.visual.ln_post(x[:, 0, :])
        #
        # if self.model.visual.proj is not None:
        #     x = x @ self.model.visual.proj
        # 获取 Prompt 的特征
        prompt_length = self.prompt_module_visual.prompt_embeddings.size(0)
        prompt_features = x[:, 1:1 + prompt_length, :]  # 提取 Prompt token 的输出

        # 对 Prompt 特征进行平均池化
        avg_prompt_features = prompt_features.mean(dim=1)  # [batch_size, d_model]

        # 使用池化后的 Prompt 特征
        x = self.model.visual.ln_post(avg_prompt_features)

        if self.model.visual.proj is not None:
            x = x @ self.model.visual.proj

        return x

    def compute_weights(self):
        """
        计算 Zero-shot 分类的类嵌入权重，支持 Prompt。
        """
        use_format = isinstance(self.templates[0], str)
        num_templates = len(self.templates)
        num_classes = len(self.classnames)

        def _process_batch(batch_classnames):
            texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in
                     self.templates]
            text_tokens = torch.stack([self.tokenize(t).to(self.device) for t in texts])

            # 使用 encode_text_with_prompt 进行特征编码
            text_features = self.encode_text_with_prompt(text_tokens)

            text_features = text_features.reshape(len(batch_classnames), num_templates, -1).mean(dim=1)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_features = text_features.T
            return text_features

        with torch.no_grad():
            if self.num_classes_per_batch:
                batched_embeds = []
                for batch in tqdm(self.batch_classes(self.classnames, self.num_classes_per_batch),
                                  desc="Computing Zero-shot Weights", unit="batch"):
                    batched_embeds.append(_process_batch(batch))
                self.zeroshot_weights = torch.cat(batched_embeds, dim=1)
            else:
                self.zeroshot_weights = _process_batch(self.classnames)

    def forward(self, images):
        """
        执行 Zero-shot 分类。
        """
        if self.zeroshot_weights is None:
            raise ValueError("Zero-shot weights not computed. Call `compute_weights` first.")

        # 使用 encode_image_with_prompt 对图像特征编码
        image_features = self.encode_image_with_prompt(images)
        logits = 100. * image_features @ self.zeroshot_weights
        return logits

    def batch_classes(self, classnames, num_classes_per_batch):
        """
        对类名分批处理。
        """
        return [classnames[i:i + num_classes_per_batch] for i in range(0, len(classnames), num_classes_per_batch)]
