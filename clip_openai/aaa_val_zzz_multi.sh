#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/ppio_net0/huggingface

# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表
datasets=(
#   "clothes"
#  "flickr30k"
#  "coco2014"
#  "pet"
#  "lexica"
#  "simpsons"
#  "patfig"
#  "wikiart"
#  "styles"
#  "kream"
#  "sketch"
  "shahnegar"
#  "clothes"
)

# 将 datasets 数组转换为逗号分隔的字符串
DATASETS=$(IFS=','; echo "${datasets[*]}")

# 检查点目录列表
ckpt_dirs=(
  "/ppio_net0/code/clip-lightning/clip_openai/ckpt-zeroshot/c-clip"
#  "/ppio_net0/code/clip-lightning/clip_openai/ckpt-zeroshot/ft-1e-5"
#  "/ppio_net0/code/clip-lightning/clip_openai/ckpt-zeroshot/ft-2e-5"
#  "/ppio_net0/code/clip-lightning/clip_openai/ckpt-zeroshot/ft-3e-5"
)

# 其他参数
CONFIG_FILE=config.yaml
ROOT_DIR=/ppio_net0/torch_ds

# 评估每个检查点目录
for CKPT_DIR in "${ckpt_dirs[@]}"; do
  echo "Evaluating with checkpoint directory: ${CKPT_DIR}"

  python cli_val_multi.py validate \
    --data.dataset_name ${DATASETS} \
    --data.batch_size 32 \
    --data.num_workers 8 \
    --data.config ${CONFIG_FILE} \
    --data.root_dir ${ROOT_DIR} \
    --model.model_name ${MODEL_NAME} \
    --model.download_root ./ \
    --model.ckpt_dir ${CKPT_DIR} \
    --trainer.accelerator gpu \
    --trainer.precision 16 \
    --trainer.max_epochs 1 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP-debug \
    --trainer.logger.name "${MODEL_NAME}-eval-$(basename ${CKPT_DIR})" \
    --trainer.logger.log_model False \
    --trainer.strategy ddp_find_unused_parameters_true

  echo "Completed evaluation for checkpoint directory: ${CKPT_DIR}"
done

echo "Completed evaluation for all checkpoint directories and datasets."
