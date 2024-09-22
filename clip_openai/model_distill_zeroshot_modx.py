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


def cognition_align(old, new):
    n, m = old.shape
    index = torch.max(old, 1)[1]  ### 找到对比矩阵中每一行的最大值索引
    for i in range(n):
        if index[i] != i:  ### 如果最大值索引不是当前行的索引值，则说明过去模型对当前该样本数据认知错误，为此将这一行替换为当前训练模型的对应矩阵行，排除其对当前训练模型的影响。
            old[i] = new[i]

    return old, new


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
            recall_eval_interval: int = 5,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = my_load(name=model_name, download_root=download_root)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.distill = True
        self.initialize_old_modules()

    def initialize_old_modules(self):
        # load task N-1 checkpoint
        if self.hparams.old_checkpoint_path is not None:
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
        parameters = [{
            "params": self.model.parameters(),
            "lr": self.hparams.lr,
            "weight_decay": self.hparams.weight_decay
        }]

        if self.distill:
            parameters.append({
                "params": self.distill_predictor.parameters(),
                "lr": self.hparams.lr,  # 可以根据需要调整学习率
                "weight_decay": self.hparams.weight_decay
            })
        else:
            # 如果不进行蒸馏，冻结参数
            for param in self.distill_predictor.parameters():
                param.requires_grad = False

        optimizer = optim.AdamW(parameters, weight_decay=self.hparams.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.lr_warmup_epochs,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=0.001 * self.hparams.lr,
            eta_min=0.001 * self.hparams.lr
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
            return clip_loss + distill_loss * 0.001
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
            return clip_loss + distill_loss * 0.001
        else:
            return clip_loss

    def on_train_start(self):
        # recall metric
        val_loader = self.trainer.datamodule.val_dataloader()
        recall_metric = self.get_recall_metrics(val_loader)
        self.log_dict(recall_metric, sync_dist=True)
        # # Zero-shot metric evaluation before training starts
        # zero_shot_loader = self.trainer.datamodule.zero_shot_dataloader()
        # zero_shot_metric = self.get_zero_shot_metrics(zero_shot_loader)
        # self.log_dict(zero_shot_metric, sync_dist=True)

    def on_validation_epoch_end(self):
        # recall metric
        if (self.current_epoch + 1) % self.hparams.recall_eval_interval == 0:
            val_loader = self.trainer.datamodule.val_dataloader()
            recall_metric = self.get_recall_metrics(val_loader)
            self.log_dict(recall_metric, sync_dist=True)

        # zero-shot metric
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
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(targets)
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

    def on_save_checkpoint(self, checkpoint):
        # 处理视觉模块中的 conv1 和 transformer
        if self.trainer.current_epoch != self.trainer.max_epochs - 1:
            pass
        elif self.trainer.current_epoch == self.trainer.max_epochs - 1:
            # conv1 = copy.deepcopy(self.model.visual.conv1)
            # self.model.visual.conv1 = conv1.merge_and_unload().conv1
            # self.model.visual.transformer.merge_and_unload()
            # self.model.transformer.merge_and_unload()

            # 仅在主进程中输出
            if self.trainer.is_global_zero:
                print('************************')

                # 创建一个新的 state_dict 用于保存权重
                new_state_dict = {}

                # 遍历当前模型的参数，处理名称
                for name, param in self.model.named_parameters():
                    # 如果参数名称中包含 'base_model.model'，则去掉
                    new_name = name.replace("base_model.model.", "")
                    new_state_dict[new_name] = param.data

                # 保存处理后的权重到检查点
                checkpoint['model'] = new_state_dict

                print('Saved model parameters with modified names:')
                for new_name in new_state_dict.keys():
                    print(new_name)

                print('************************')

                # 比较新模型参数名与旧模型参数名
                print('Parameter Comparison with Old Model:')
                for (new_name, param), (old_name, old_param) in zip(new_state_dict.items(),
                                                                    self.model_old.named_parameters()):
                    if new_name != old_name:
                        print(f"New: {new_name} | Old: {old_name}")

                print('************************')
