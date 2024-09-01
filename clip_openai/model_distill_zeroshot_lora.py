import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from model_openai import my_load
import copy

from zero_shot.zero_shot_classifier import ZeroShotClassifier
from model_openai import SimpleTokenizer
from zero_shot.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from timm.utils import accuracy
from tqdm import tqdm

from peft import get_peft_model, LoraConfig, TaskType
from transformers.pytorch_utils import Conv1D


def find_target_modules(model):
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            target_modules.append(name)
    return target_modules


def get_lora_model(model):
    # Define the target modules where LoRA should be applied
    target_modules = find_target_modules(model)

    # Initialize LoRA configuration with target modules
    lora_config = LoraConfig(
        inference_mode=False,
        r=16,  # Rank of the low-rank decomposition
        lora_alpha=32,  # Scaling factor
        task_type=TaskType.SEQ_CLS,  # Task type
        lora_dropout=0.1,  # Dropout rate for LoRA
        target_modules=target_modules  # Specify the target modules
    )

    # Apply LoRA to the model
    lora_model = get_peft_model(model, lora_config)

    return lora_model


def filter_layers(target_modules, layers_to_include):
    """
    根据指定的层索引过滤目标模块。

    参数:
    - target_modules: List[str]，find_target_modules 函数返回的模块名称列表。
    - layers_to_include: List[int]，需要包含的层索引列表。

    返回:
    - filtered_modules: List[str]，经过过滤后的模块名称列表。
    """
    filtered_modules = []
    for layer_index in layers_to_include:
        # 构造层名字符串，例如 'resblocks.1'
        layer_name = f'resblocks.{layer_index}'
        # 过滤出包含该层名的模块
        filtered_modules.extend([name for name in target_modules if layer_name in name])

    return filtered_modules


def get_lora_model_vision(model):
    # Define the target modules where LoRA should be applied
    target_modules = find_target_modules(model)
    # print(target_modules)
    # target_modules = filter_layers(target_modules, [0, 1, 2, 3, 4, 5, 6])
    # print(target_modules)

    # Initialize LoRA configuration with target modules
    lora_config = LoraConfig(
        inference_mode=False,
        r=256,  # Rank of the low-rank decomposition
        lora_alpha=128,  # Scaling factor
        task_type='vision',  # Task type
        lora_dropout=0.,  # Dropout rate for LoRA
        target_modules=target_modules  # Specify the target modules
        # target_modules='all-linear'  # Specify the target modules
    )

    # Apply LoRA to the model
    lora_model = get_peft_model(model, lora_config)

    return lora_model


def get_lora_model_text(model):
    # Define the target modules where LoRA should be applied
    target_modules = find_target_modules(model)
    print(target_modules)

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


# def get_lora_model2(model):
#     # Define the target modules where LoRA should be applied
#     target_modules = find_target_modules(model)
#
#     # Initialize LoRA configuration with target modules
#     lora_config = LoraConfig(
#         inference_mode=False,
#         r=16,  # Rank of the low-rank decomposition
#         lora_alpha=32,  # Scaling factor
#         task_type='vision',  # Task type
#         lora_dropout=0.1,  # Dropout rate for LoRA
#         target_modules=target_modules  # Specify the target modules
#     )
#
#     # Apply LoRA to the model
#     lora_model = get_peft_model(model, lora_config)
#
#     return lora_model


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

    # def forward(self, x):
    #     residual = x  # 保存输入以用于残差连接
    #     x = self.pd_linear1(x)
    #     x = self.pd_batch_norm(x)
    #     x = self.pd_relu(x)
    #     x = self.pd_linear2(x)
    #     x += residual  # 加上残差连接
    #     return x


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'RN50',
            download_root: str = None,
            projection_dims: int = 1024,
            temperature: float = 1.0,
            weight_decay: float = 0.0,
            lr: float = 1e-3,
            lr_warmup_epochs: int = 5,
            batch_size: int = 64,
            old_checkpoint_path: str = None,
            current_task: int = 0,
            batch_size_zs: int = 256,
            zero_shot_eval_interval: int = 5,
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

        # self.model = get_lora_model(self.model)
        self.model.transformer = get_lora_model_text(self.model.transformer)
        self.model.visual = get_lora_model_vision(self.model.visual)
        print('********************************************')
        print(self.model)
        # print(self.model.visual.named_parameters())

        for name, param in self.model.named_parameters():
            print(name)

        # # 定义需要冻结的完整参数名
        # exact_layers_to_freeze = [
        #     "base_model.model.class_embedding",
        #     "base_model.model.positional_embedding",
        #     "base_model.model.proj",
        #     "base_model.model.conv1.weight",
        #     "base_model.model.ln_pre.weight",
        #     "base_model.model.ln_pre.bias"
        # ]
        #
        # # 冻结参数
        # for name, param in self.model.visual.named_parameters():
        #     # 如果参数名在完整匹配的列表中，或包含 "resblocks.0" 到 "resblocks.5"，则冻结
        #     if name in exact_layers_to_freeze or any(f"resblocks.{i}" in name for i in range(9)):
        #         param.requires_grad = False

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

    def initialize_old_modules(self):
        self.model_old = copy.deepcopy(self.model)
        # Set requires_grad to False for all parameters in the old modules
        for param in self.model_old.parameters():
            param.requires_grad = False

        # distill project
        distill_proj_hidden_dim = 2048
        # self.distill_predictor = nn.Sequential(
        #     nn.Linear(self.hparams.projection_dims, distill_proj_hidden_dim),
        #     nn.BatchNorm1d(distill_proj_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(distill_proj_hidden_dim, self.hparams.projection_dims),
        # )
        self.distill_predictor = DistillPredictor(
            projection_dims=self.hparams.projection_dims,
            distill_proj_hidden_dim=distill_proj_hidden_dim
        )
        # self.distill_predictor = get_lora_model2(self.distill_predictor)
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
        parameters = [{
            "params": self.model.parameters(),
            "lr": self.hparams.lr,
            "weight_decay": self.hparams.weight_decay
        }]

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

    def simclr_distill_loss_func(
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

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        self.log("train/clip_loss", clip_loss, sync_dist=True)

        if self.distill:
            frozen_z1, frozen_z2 = self.forward_old(batch)
            p1 = self.distill_predictor(image_embeddings)
            p2 = self.distill_predictor(text_embeddings)

            distill_loss = (
                                   self.simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2)
                                   + self.simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2)
                           ) / 2

            self.log("train/distill_loss", distill_loss, sync_dist=True)
            return clip_loss + distill_loss
        else:
            return clip_loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        self.log("val/clip_loss", clip_loss, sync_dist=True)
        self.val_img_feats.append(image_embeddings)
        self.val_text_feats.append(text_embeddings)

        if self.distill:
            frozen_z1, frozen_z2 = self.forward_old(batch)
            p1 = self.distill_predictor(image_embeddings)
            p2 = self.distill_predictor(text_embeddings)

            distill_loss = (
                                   self.simclr_distill_loss_func(p1, p2, frozen_z1, frozen_z2)
                                   + self.simclr_distill_loss_func(frozen_z1, frozen_z2, p1, p2)
                           ) / 2
            self.log("val/distill_loss", distill_loss, sync_dist=True)
            return clip_loss + distill_loss
        else:
            return clip_loss

    def on_train_start(self):
        # Zero-shot metric evaluation before training starts
        zero_shot_loader = self.trainer.datamodule.zero_shot_dataloader()
        zero_shot_metric = self.get_zero_shot_metrics(zero_shot_loader)
        self.log_dict(zero_shot_metric, sync_dist=True)

    def on_validation_epoch_end(self):
        all_image_features = torch.cat(self.val_img_feats)
        all_text_features = torch.cat(self.val_text_feats)
        val_metrics = self.get_clip_metrics_cpu(
            image_features=all_image_features,
            text_features=all_text_features,
            logit_scale=self.model.logit_scale.exp(),
        )
        self.log_dict(val_metrics, sync_dist=True)
        self.val_img_feats.clear()
        self.val_text_feats.clear()

        # zero-shot metric
        if (self.current_epoch + 1) % self.hparams.zero_shot_eval_interval == 0:
            zero_shot_loader = self.trainer.datamodule.zero_shot_dataloader()
            zero_shot_metric = self.get_zero_shot_metrics(zero_shot_loader)
            self.log_dict(zero_shot_metric, sync_dist=True)

    def get_clip_metrics(self, image_features, text_features, logit_scale=1.0):
        metrics = {}
        logits_per_image = (logit_scale * image_features @ text_features.t())
        logits_per_text = logits_per_image.t()
        logits = {"val/image_to_text": logits_per_image, "val/text_to_image": logits_per_text}
        ground_truth = torch.arange(len(text_features)).view(-1, 1).to(self.device)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True).to(self.device)
            preds = torch.where(ranking == ground_truth)[1]
            for k in [1]:
                metrics[f"{name}_R@{k}"] = (preds < k).float().mean() * 100  # Convert recall to percentage

        return metrics

    def get_clip_metrics_cpu(self, image_features, text_features, logit_scale=1.0):
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
                metrics[f"{name}_R@{k}"] = np.mean(preds < k) * 100  # Convert recall to percentage

        return metrics

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
