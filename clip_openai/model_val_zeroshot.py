import os
from model_openai import SimpleTokenizer
import torch
from lightning import LightningModule
from model_openai import my_load
from zero_shot.zero_shot_classifier import ZeroShotClassifier
from timm.utils import accuracy
from tqdm import tqdm
import wandb


def get_metadata(dataset_name):
    if dataset_name == "cifar100":
        from zero_shot.zero_shot_metadata_cifar100 import classes, templates
        return classes, templates
    elif dataset_name == "stl10":
        from zero_shot.zero_shot_metadata_stl10 import classes, templates
        return classes, templates
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'RN50',
            download_root: str = None,
            ckpt_dir: str = None,
            batch_size: int = 32,

            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = my_load(name=model_name, download_root=download_root)
        self.tokenizer = SimpleTokenizer()

    def load_task_checkpoint(self, checkpoint_path):
        checkpoint_path = os.path.join(self.hparams.ckpt_dir, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model'], strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")

    def get_zero_shot_metrics(self, dataloader, classnames, templates):
        """评估 zero-shot 性能"""
        self.zero_shot_classifier = ZeroShotClassifier(
            model=self.model,
            tokenizer=self.tokenizer,
            classnames=classnames,
            templates=templates,
            num_classes_per_batch=self.hparams.batch_size,
        ).to(self.device)

        self.zero_shot_classifier.compute_weights()

        top1, n = 0., 0.
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Zero-shot Evaluating", unit="batch"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.zero_shot_classifier(images)
                acc1, _ = accuracy(logits, targets, topk=(1, 5))
                top1 += acc1.item() * images.size(0)
                n += images.size(0)

        top1 = top1 / n
        del self.zero_shot_classifier  # 释放内存
        return top1

    def validation_step(self, batch, *args, **kwargs):
        self.log("val/clip_loss", 0., sync_dist=True)

        return 0

    def on_validation_epoch_end(self):
        dataset_all = self.trainer.datamodule.dataset_name
        task_checkpoints = [
            "task1.ckpt", "task2.ckpt", "task3.ckpt", "task4.ckpt",
            "task5.ckpt", "task6.ckpt", "task7.ckpt", "task8.ckpt"
        ]

        all_metrics = []

        for dataset_name in dataset_all:
            classnames, templates = get_metadata(dataset_name)
            zero_shot_loader = self.trainer.datamodule.zero_shot_dataloader(dataset_name)

            metrics_row = {"dataset": dataset_name}

            for task_idx, task_ckpt in enumerate(task_checkpoints):
                self.load_task_checkpoint(task_ckpt)
                self.model.eval()

                top1_accuracy = self.get_zero_shot_metrics(zero_shot_loader, classnames, templates)
                metrics_row[f"task{task_idx + 1}"] = top1_accuracy

            all_metrics.append(metrics_row)

        self.log_metrics_to_wandb(all_metrics)

    def log_metrics_to_wandb(self, all_metrics):
        table_data = []
        for row in all_metrics:
            table_data.append([row["dataset"]] + [row[f"task{i}"] for i in range(1, 9)])

        table = wandb.Table(columns=["Dataset", "task1", "task2", "task3", "task4", "task5", "task6", "task7", "task8"],
                            data=table_data)
        wandb.log({"zero_shot_results": table})
