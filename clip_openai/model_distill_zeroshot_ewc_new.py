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
from zero_shot.zero_shot_metadata_imagenet import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from timm.utils import accuracy
from tqdm import tqdm


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

        # EWC parameters
        self.use_ewc = True
        self.ewc_lambda = 0.4  # You can adjust this value
        self.fisher_information = {}
        self.params_old = {}

        self.initialize_old_modules()

    def initialize_old_modules(self):
        # Load task N-1 checkpoint
        if self.hparams.old_checkpoint_path is not None:
            checkpoint = torch.load(self.hparams.old_checkpoint_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model'], strict=True)
            print("Model weights loaded successfully and old parts copied.")

        self.model_old = copy.deepcopy(self.model)
        # No need to set requires_grad=False; we need gradients for EWC

    def forward(self, inputs):
        image_features = self.model.encode_image(inputs["image"])
        text_features = self.model.encode_text(inputs["caption"])
        return image_features, text_features

    def _compute_losses(self, image_features, text_features):
        # Normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Shape = [global_batch_size, global_batch_size]
        labels = torch.arange(len(logits_per_image)).to(self.device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        loss = (image_loss + text_loss) / 2

        return loss

    def configure_optimizers(self):
        parameters = [
            {
                "params": self.model.visual.parameters(),  # Visual part with its own learning rate
                "lr": self.hparams.lr
            },
            {
                "params": [param for name, param in self.model.named_parameters() if "visual" not in name],
                "lr": self.hparams.lr_text,
                "weight_decay": self.hparams.weight_decay
            }
        ]

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

    def compute_fisher_information(self, dataloader):
        self.model_old.eval()
        # Initialize the fisher_information dict
        fisher_information = {}
        param_names = [name for name, param in self.model_old.named_parameters() if param.requires_grad]
        for name in param_names:
            fisher_information[name] = torch.zeros_like(param)

        n_samples = 0
        self.model_old.to(self.device)

        for batch in tqdm(dataloader, desc="Computing Fisher Information"):
            self.model_old.zero_grad()
            images = batch["image"].to(self.device)
            captions = batch["caption"].to(self.device)
            # Forward pass without torch.no_grad() to compute gradients
            image_features = self.model_old.encode_image(images)
            text_features = self.model_old.encode_text(captions)
            loss = self._compute_losses(image_features, text_features)
            loss.backward()
            n_samples += 1

            for name, param in self.model_old.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad.detach() ** 2

        # Average over number of samples
        for name in fisher_information:
            fisher_information[name] = fisher_information[name] / n_samples

        self.fisher_information = fisher_information
        # Store old parameters
        self.params_old = {name: param.clone().detach() for name, param in self.model_old.named_parameters()}

    def ewc_loss(self):
        loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Get the old parameter
                theta_i_old = self.params_old[name].to(self.device)
                # Get the Fisher Information
                F_i = self.fisher_information[name].to(self.device)
                # Compute the loss
                loss += (F_i * (param - theta_i_old).pow(2)).sum()
        return (self.ewc_lambda / 2) * loss

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        self.log("train/clip_loss", clip_loss, sync_dist=True)

        if self.use_ewc:
            ewc_loss = self.ewc_loss()
            self.log("train/distill_loss", ewc_loss, sync_dist=True)
            return clip_loss + ewc_loss
        else:
            return clip_loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        clip_loss = self._compute_losses(image_embeddings, text_embeddings)
        self.log("val/clip_loss", clip_loss, sync_dist=True)

        if self.use_ewc:
            ewc_loss = self.ewc_loss()
            self.log("val/distill_loss", ewc_loss, sync_dist=True)
            return clip_loss + ewc_loss
        else:
            return clip_loss

    def on_train_start(self):
        # Compute Fisher Information Matrix
        if self.use_ewc:
            val_loader = self.trainer.datamodule.val_dataloader()
            self.compute_fisher_information(val_loader)
        # Recall metric
        val_loader = self.trainer.datamodule.val_dataloader()
        recall_metric = self.get_recall_metrics(val_loader)
        self.log_dict(recall_metric, sync_dist=True)

    def on_validation_epoch_end(self):
        # Recall metric
        if (self.current_epoch + 1) % self.hparams.recall_eval_interval == 0:
            val_loader = self.trainer.datamodule.val_dataloader()
            recall_metric = self.get_recall_metrics(val_loader)
            self.log_dict(recall_metric, sync_dist=True)

        # Zero-shot metric
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
        # Handle visual module's conv1 and transformer
        if self.trainer.current_epoch != self.trainer.max_epochs - 1:
            pass
        elif self.trainer.current_epoch == self.trainer.max_epochs - 1:
            # Only output in the main process
            if self.trainer.is_global_zero:
                print('************************')

                # Create a new state_dict for saving weights
                new_state_dict = {}

                # Iterate over current model parameters and process names
                for name, param in self.model.named_parameters():
                    # Remove 'base_model.model' from parameter names if present
                    new_name = name.replace("base_model.model.", "")
                    new_state_dict[new_name] = param.data

                # Save processed weights to checkpoint
                checkpoint['model'] = new_state_dict

                print('Saved model parameters with modified names:')
                for new_name in new_state_dict.keys():
                    print(new_name)

                print('************************')

                # Compare new model parameter names with old model parameter names
                print('Parameter Comparison with Old Model:')
                for (new_name, param), (old_name, old_param) in zip(new_state_dict.items(),
                                                                    self.model_old.named_parameters()):
                    if new_name != old_name:
                        print(f"New: {new_name} | Old: {old_name}")

                print('************************')
