import numpy as np
import torch
from lightning import LightningModule
from model_openai import my_load
from model_openai import SimpleTokenizer
from tqdm import tqdm
import wandb
from packaging import version
from metric import i2t, t2i
import os


class CLIPDualEncoderModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'RN50',
            download_root: str = None,
            batch_size: int = 64,
            max_length: int = 77,
            ckpt_dir: str = None,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = my_load(name=model_name, download_root=download_root)
        self.tokenizer = SimpleTokenizer()
        self.max_length = max_length

    def load_task_checkpoint(self, checkpoint_path):
        """加载每个任务的 checkpoint"""
        checkpoint_path = os.path.join(self.hparams.ckpt_dir, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model'], strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")

    def tokenize(self, text):
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        if version.parse(torch.__version__) < version.parse("1.8.0"):
            result = torch.zeros(self.max_length, dtype=torch.long)
        else:
            result = torch.zeros(self.max_length, dtype=torch.int)

        if len(tokens) <= self.max_length:
            result[:len(tokens)] = torch.tensor(tokens)
        else:
            result[:self.max_length] = torch.tensor(tokens)[:self.max_length]
        return result

    def recall_score_1caption(self, image_features, text_features, logit_scale=1.0):
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

    def recall_score_5caption(self, image_features, text_features, caps_per_image=5):
        i2t_r1 = i2t(
            images=image_features.cpu().numpy(),
            captions=text_features.cpu().numpy(),
            caps_per_image=caps_per_image
        )
        t2i_r1 = t2i(
            images=image_features.cpu().numpy(),
            captions=text_features.cpu().numpy(),
            caps_per_image=caps_per_image
        )
        metrics = {
            "val/image_to_text_R@1": i2t_r1,
            "val/text_to_image_R@1": t2i_r1
        }

        return metrics

    def get_recall_metrics_5caption(self, dataloader, num_caption=5):
        val_img_feats = []
        val_text_feats = []

        with torch.no_grad():
            for inputs in tqdm(dataloader, desc="Recall Evaluating", unit="batch"):
                images = inputs["image"].to(self.device)
                batch_size = images.size(0)

                images = images.repeat_interleave(num_caption, dim=0)  # [batch_size * num_caption, C, H, W]

                # 展开 captions 列表
                texts = [inputs["text"][i][j] for j in range(batch_size) for i in range(num_caption)]
                texts = [self.tokenize(t).to(self.device) for t in texts]
                texts = torch.stack(texts)

                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                val_img_feats.append(image_features)
                val_text_feats.append(text_features)

        all_image_features = torch.cat(val_img_feats)
        all_text_features = torch.cat(val_text_feats)

        metrics = self.recall_score_5caption(
            image_features=all_image_features,
            text_features=all_text_features,
            caps_per_image=num_caption
        )

        return metrics

    def get_recall_metrics_1caption(self, dataloader):
        val_img_feats = []
        val_text_feats = []

        with torch.no_grad():
            for inputs in tqdm(dataloader, desc="Recall Evaluating", unit="batch"):
                images = inputs["image"].to(self.device)
                texts = [self.tokenize(t).to(self.device) for t in inputs["text"]]
                texts = torch.stack(texts)
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                val_img_feats.append(image_features)
                val_text_feats.append(text_features)

        all_image_features = torch.cat(val_img_feats)
        all_text_features = torch.cat(val_text_feats)

        metrics = self.recall_score_1caption(
            image_features=all_image_features,
            text_features=all_text_features,
            logit_scale=self.model.logit_scale.exp(),
        )

        return metrics

    def on_validation_epoch_end(self):
        dataset_all = self.trainer.datamodule.dataset_name
        task_checkpoints = [
            None,  # task0，不加载任何 checkpoint
            "task1.ckpt", "task2.ckpt", "task3.ckpt", "task4.ckpt",
            "task5.ckpt", "task6.ckpt", "task7.ckpt", "task8.ckpt"
        ]

        all_metrics = []

        # 循环数据集
        for dataset_name in dataset_all:
            self.model = my_load(name=self.hparams.model_name, download_root=self.hparams.download_root)
            self.model.to(self.device)
            val_loader = self.trainer.datamodule.val_dataloader(dataset_name)

            # 创建保存结果的字典
            metrics_row = {"dataset": dataset_name}

            # 循环不同任务的 checkpoint
            for task_idx, task_ckpt in enumerate(task_checkpoints):
                if task_ckpt is not None:
                    # 加载 checkpoint
                    self.load_task_checkpoint(task_ckpt)
                else:
                    print("Evaluating task0 with the initialized model (no checkpoint loaded)")

                self.model.eval()

                # 选择合适的 recall 评估方法
                if dataset_name in ['flickr30k', 'coco2014']:
                    recall_metric = self.get_recall_metrics_5caption(val_loader, num_caption=5)
                else:
                    recall_metric = self.get_recall_metrics_1caption(val_loader)

                # 将 recall 结果保存到当前任务中
                metrics_row[f"task{task_idx}_image2text_R@1"] = recall_metric.get("val/image_to_text_R@1", 0)
                metrics_row[f"task{task_idx}_text2image_R@1"] = recall_metric.get("val/text_to_image_R@1", 0)

            all_metrics.append(metrics_row)

        # 将指标记录到 W&B
        self.log_metrics_to_wandb(all_metrics)

    def log_metrics_to_wandb(self, all_metrics):
        table_data = []
        for row in all_metrics:
            # 创建包含 dataset 和各任务指标的数据行
            table_data.append([row["dataset"]] +
                              [row[f"task{i}_image2text_R@1"] for i in range(9)] +
                              [row[f"task{i}_text2image_R@1"] for i in range(9)])

        # 创建包含所有任务名称的列名
        columns = (["Dataset"] +
                   [f"task{i}_image2text_R@1" for i in range(9)] +
                   [f"task{i}_text2image_R@1" for i in range(9)])
        table = wandb.Table(columns=columns, data=table_data)
        wandb.log({"recall_results": table})

    def validation_step(self, batch, *args, **kwargs):
        pass
