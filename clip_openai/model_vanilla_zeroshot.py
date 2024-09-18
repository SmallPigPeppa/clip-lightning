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
from zero_shot.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from timm.utils import accuracy
from tqdm import tqdm
import copy
import random

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
        self.initialize_old_modules()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.val_img_feats = []
        self.val_text_feats = []

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

    # def forward(self, inputs):
    #     image_features = self.model.encode_image(inputs["image"])
    #     text_features = self.model.encode_text(inputs["caption"])
    #     return image_features, text_features

    # def forward(self, inputs, random_select=True):
    #     image_features = self.model.encode_image(inputs["image"])
    #
    #     if inputs.get("multi_caption", False):
    #         if random_select:
    #             # Randomly select one caption
    #             selected_caption = random.choice(inputs["caption"])
    #             text_features = self.model.encode_text(selected_caption)
    #         else:
    #             # Use all captions if random_select is False
    #             text_features = [self.model.encode_text(caption) for caption in inputs["caption"]]
    #     else:
    #         # Single caption scenario
    #         text_features = self.model.encode_text(inputs["caption"])
    #
    #     return image_features, text_features

    def forward(self, inputs, random_select=True):
        image_features = self.model.encode_image(inputs["image"])

        # Check the 'multi_caption' flag for the first item in the batch
        if inputs["multi_caption"][0]:
            if random_select:
                # Randomly select one caption for each item in the batch
                # selected_captions = [random.choice(captions) for captions in inputs["caption"]]
                # selected_captions = torch.cat(selected_captions, dim=0)
                selected_caption = random.choice(inputs["caption"])
                text_features = self.model.encode_text(selected_caption)
            else:
                # Use all captions for each item in the batch if random_select is False
                text_features = [self.model.encode_text(captions) for captions in inputs["caption"]]
        else:
            # Single caption scenario
            text_features = self.model.encode_text(inputs["caption"])

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

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        self.log("train/clip_loss", clip_loss, sync_dist=True)

        return clip_loss

    # def validation_step(self, batch, *args, **kwargs):
    #     image_embeddings, text_embeddings = self.forward(batch, random_select=False)
    #     clip_loss = self._compute_losses(image_embeddings, text_embeddings)
    #     self.log("val/clip_loss", clip_loss, sync_dist=True)
    #     self.val_img_feats.append(image_embeddings)
    #     self.val_text_feats.append(text_embeddings)
    #
    #     return clip_loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch, random_select=False)
        image_embeddings_one, text_embeddings_one = self.forward(batch, random_select=True)
        clip_loss = self._compute_losses(image_embeddings_one, text_embeddings_one)
        self.log("val/clip_loss", clip_loss, sync_dist=True)
        self.val_img_feats.append(image_embeddings)
        self.val_text_feats.append(text_embeddings)

        return clip_loss

    # def on_train_start(self):
    #     # Zero-shot metric evaluation before training starts
    #     zero_shot_loader = self.trainer.datamodule.zero_shot_dataloader()
    #     zero_shot_metric = self.get_zero_shot_metrics(zero_shot_loader)
    #     self.log_dict(zero_shot_metric, sync_dist=True)

    def on_validation_epoch_end(self):
        import pdb;pdb.set_trace()
        all_image_features = torch.cat(self.val_img_feats)
        # all_text_features = torch.cat(self.val_text_feats)

        if isinstance(self.val_text_feats[0], torch.Tensor) and len(self.val_text_feats[0].shape) == 2:
            # Use torch.cat to concatenate B, D tensors along the 0th dimension
            all_text_features=torch.cat(self.val_text_feats, dim=0)
        else:
            # # First, stack each K-length list of B, D tensors along dimension 1 to get B, K, D
            # concatenated_tensors = [torch.stack(k_list, dim=1) for k_list in self.val_text_feats]
            # # Then, concatenate all B, K, D tensors along dimension 0 to get SB, K, D
            # all_text_features=torch.cat(concatenated_tensors, dim=0)
            val_text_feats_tensor = torch.tensor(self.val_text_feats).to(self.device)
            val_text_feats_tensor = val_text_feats_tensor.permute(0, 2, 1, 3)  # (S, B, 5, 512)
            all_text_features = val_text_feats_tensor.flatten(0, 1)  # 直接合并前两维，结果是 (SB, 5, 512)


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

    # def get_clip_metrics_cpu(self, image_features, text_features, logit_scale=1.0):
    #     metrics = {}
    #     logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    #     logits_per_text = logits_per_image.t().detach().cpu()
    #
    #     logits = {"val/image_to_text": logits_per_image, "val/text_to_image": logits_per_text}
    #     ground_truth = torch.arange(len(text_features)).view(-1, 1)
    #
    #     for name, logit in logits.items():
    #         ranking = torch.argsort(logit, descending=True)
    #         preds = torch.where(ranking == ground_truth)[1]
    #         preds = preds.detach().cpu().numpy()
    #         for k in [1]:
    #             metrics[f"{name}_R@{k}"] = np.mean(preds < k) * 100  # Convert recall to percentage
    #
    #     return metrics

    def get_clip_metrics_cpu(self, image_features, text_features, logit_scale=1.0):
        metrics = {}

        # Handle case where text_features has shape (N, M, D) - multiple captions for each image
        if len(text_features.shape) == 3:
            N, M, D = text_features.shape
            # Reshape text_features to (N * M, D) to compute logits for all image-caption pairs
            text_features = text_features.view(-1, D)

            # Calculate logits per image
            logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
            logits_per_image = logits_per_image.view(N, M, N)  # Reshape to (N, M, N) - (images, captions, images)

            # Get the max logit across the multiple captions for each image
            logits_per_image, _ = torch.max(logits_per_image, dim=1)  # Max across M captions

        else:
            # Standard case where text_features is (N, D)
            logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()

        logits_per_text = logits_per_image.t().detach().cpu()

        # Compute recall metrics
        logits = {"val/image_to_text": logits_per_image, "val/text_to_image": logits_per_text}
        ground_truth = torch.arange(len(image_features)).view(-1, 1)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == ground_truth)[1]
            preds = preds.detach().cpu().numpy()

            for k in [1]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k) * 100  # Convert recall to percentage

        return metrics

    # def get_clip_metrics_cpu(self, image_features, text_features, logit_scale=1.0):
    #     metrics = {}
    #
    #     # 确定文本特征的维度
    #     if text_features.dim() == 3:  # N, M, D
    #         # 用于广播，计算每个图片与多个文本间的logits
    #         image_features_exp = image_features.unsqueeze(1)  # N, 1, D
    #         logits_per_image = logit_scale * (image_features_exp @ text_features.transpose(-2, -1)).squeeze(1)
    #     else:  # N, D
    #         logits_per_image = logit_scale * (image_features @ text_features.t())
    #
    #     logits_per_image = logits_per_image.detach().cpu()
    #     logits_per_text = logits_per_image.t().detach().cpu()
    #
    #     logits = {"val/image_to_text": logits_per_image, "val/text_to_image": logits_per_text}
    #     ground_truth = torch.arange(len(image_features)).view(-1, 1)
    #
    #     for name, logit in logits.items():
    #         ranking = torch.argsort(logit, descending=True)
    #         preds = torch.where(ranking == ground_truth)[1]
    #         preds = preds.detach().cpu().numpy()
    #         for k in [1]:  # 可以根据需要调整这里的k值
    #             metrics[f"{name}_R@{k}"] = np.mean(preds < k) * 100  # Convert recall to percentage
    #
    #     return metrics

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