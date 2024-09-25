import numpy as np
import torch
from lightning import LightningModule
from model_openai import my_load
from zero_shot.zero_shot_classifier import ZeroShotClassifier
from model_openai import SimpleTokenizer
from zero_shot.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from timm.utils import accuracy
from tqdm import tqdm
from typing import Union, List
import pandas as pd
import wandb


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'RN50',
            download_root: str = None,
            batch_size: int = 64,
            batch_size_zs: int = 256,
            old_checkpoint_path: str = None,
            result_path: str = 'metrics_evaluation.xlsx',
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = my_load(name=model_name, download_root=download_root)
        self.initialize_old_modules()

    def initialize_old_modules(self):
        if self.hparams.old_checkpoint_path is not None:
            if ',' in self.hparams.old_checkpoint_path:
                self.hparams.old_checkpoint_path = self.hparams.old_checkpoint_path.split(',')
                # 初始化一个字典来存储所有检查点的参数和计数
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

                # 计算均值
                for k in avg_params.keys():
                    avg_params[k] /= count

                # 加载均值参数
                self.model.load_state_dict(avg_params, strict=True)
                print("Model weights loaded successfully and old parts averaged.")
            else:
                checkpoint = torch.load(self.hparams.old_checkpoint_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(checkpoint['model'], strict=True)
                print("Model weights loaded successfully and old parts copied.")

    def validation_step(self, batch, *args, **kwargs):
        self.log("val/clip_loss", 0., sync_dist=True)

        return 0

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

    def on_validation_epoch_end(self):
        dataset_all = self.trainer.datamodule.dataset_name  # assuming dataset names are in the datamodule
        all_metrics = []

        for idx, dataset_name in enumerate(dataset_all):
            val_loader = self.trainer.datamodule.val_dataloader(dataset_name)
            recall_metric = self.get_recall_metrics(val_loader)
            self.log_dict(recall_metric, sync_dist=True)

            metrics_row = {
                "dataset": dataset_name,
                "image2text_recall": recall_metric.get("val/image_to_text_R@1", 0),
                "text2image_recall": recall_metric.get("val/text_to_image_R@1", 0)
            }

            if idx == 0:  # Zero-shot evaluation only for the first dataset
                zero_shot_loader = self.trainer.datamodule.zero_shot_dataloader()
                zero_shot_metric = self.get_zero_shot_metrics(zero_shot_loader)
                metrics_row["zero_shot_top1"] = zero_shot_metric.get("zero_shot/top1_accuracy", 0)
                metrics_row["zero_shot_top5"] = zero_shot_metric.get("zero_shot/top5_accuracy", 0)
                self.log_dict(zero_shot_metric, sync_dist=True)

            all_metrics.append(metrics_row)

        # Save to Excel
        self.save_metrics_to_excel(all_metrics)

        # Log the table to W&B
        self.log_metrics_to_wandb(all_metrics)

    # def save_metrics_to_excel(self, metrics):
    #     df = pd.DataFrame(metrics)
    #     df.to_excel(self.hparams.result_path, index=False)
    #     print("Metrics saved to Excel.")

    def save_metrics_to_excel(self, metrics):
        # 创建 DataFrame
        df = pd.DataFrame(metrics)

        # 确保按照自定义的 dataset_name 列表顺序
        dataset_order = self.trainer.datamodule.dataset_name  # 假设 dataset_name 是你定义的顺序列表
        df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)

        # 转换为长格式（适用于 pivot 操作），将每个指标作为一列
        df_melt = pd.melt(df, id_vars=["dataset"],
                          value_vars=["image2text_recall", "text2image_recall", "zero_shot_top1", "zero_shot_top5"],
                          var_name="metric", value_name="value")

        # 进行 pivot 操作，数据集为列，metric 为行
        df_pivot = df_melt.pivot(index="metric", columns="dataset", values="value")

        # 按照自定义顺序排列列
        df_pivot = df_pivot[dataset_order]

        # 保存为 Excel 文件
        df_pivot.to_excel(self.hparams.result_path, index=True)
        print("Metrics saved to Excel.")

    # def log_metrics_to_wandb(self, metrics):
    #     table = wandb.Table(
    #         columns=["Dataset", "Image2Text Recall", "Text2Image Recall", "Zero-shot Top1", "Zero-shot Top5"])
    #
    #     for metric in metrics:
    #         table.add_data(
    #             metric.get("dataset"),
    #             metric.get("image2text_recall"),
    #             metric.get("text2image_recall"),
    #             metric.get("zero_shot_top1", 0),
    #             metric.get("zero_shot_top5", 0)
    #         )
    #
    #     wandb.log({"metrics_table": table})

    def log_metrics_to_wandb(self, metrics):
        # 创建表格，列为数据集名，行代表 metric
        columns = ["Metric"] + [metric.get("dataset") for metric in metrics]
        table = wandb.Table(columns=columns)

        # 将每个 metric 作为行添加，行首为 metric 名称，后面为不同数据集的值
        metrics_list = ["image2text_recall", "text2image_recall", "zero_shot_top1", "zero_shot_top5"]
        for metric_name in metrics_list:
            row_data = [metric_name]
            for metric in metrics:
                row_data.append(metric.get(metric_name, 0))
            table.add_data(*row_data)

        wandb.log({"metrics_table": table})

