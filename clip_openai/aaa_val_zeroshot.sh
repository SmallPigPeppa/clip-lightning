#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/ppio_net0/huggingface
# 模型名称
MODEL_NAME=ViT-B/16

datasets=(
  "cifar100"
  "stl10"
)
DATASETS=$(
  IFS=','
  echo "${datasets[*]}"
)


# 拼接检查点
CKPT_DIR="/ppio_net0/code/clip-lightning/clip_openai/ckpt-cl"
# 其他参数
CONFIG_FILE=config.yaml
ROOT_DIR=/ppio_net0/torch_ds

# 评估所有数据集
python cli_val_zeroshot.py validate \
  --data.dataset_name ${DATASETS} \
  --data.max_length 77 \
  --data.batch_size 128 \
  --data.batch_size_zs 32 \
  --data.num_workers 8 \
  --data.config ${CONFIG_FILE} \
  --data.root_dir ${ROOT_DIR} \
  --model.result_path metrics_evaluation.xlsx \
  --model.model_name ${MODEL_NAME} \
  --model.download_root ./ \
  --model.ckpt_dir ${CKPT_DIR} \
  --trainer.accelerator gpu \
  --trainer.precision 16 \
  --trainer.max_epochs 1 \
  --trainer.log_every_n_steps 1 \
  --trainer.logger WandbLogger \
  --trainer.logger.project CLIP-debug \
  --trainer.logger.name ${MODEL_NAME}-eval-all \
  --trainer.logger.log_model False \
  --trainer.strategy ddp_find_unused_parameters_true

echo "Completed evaluation for all datasets"
