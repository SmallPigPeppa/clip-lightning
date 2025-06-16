#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/mnt/hdfs/byte_content_security/user/liuwenzhuo/hf_cache
ROOT_DIR=/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets
PROJECT=CLIP-MSUN

# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表

DATASETS=("flickr30k")
#DATASETS=("coco2014")


# 数据集学习率映射
declare -A DATASET_LR_MAP=(
  ["flickr30k"]=2e-5
  ["coco2014"]=2e-5
)


# 脚本方法映射
declare -A METHOD_MAP=(
  ["fixedres"]="cli_fixedres.py"
)

# 其他参数
CONFIG_FILE=config.yaml
ZERO_SHOT_EVAL_INTERVAL=40
MAX_EPOCHS=40
BATCH_SIZE=128
BATCH_SIZE_ZS=32
NUM_WORKERS=8

# 函数：运行训练
run_training() {
  local method=$1
  local dataset_name=$2
  local lr=$3

  python ${METHOD_MAP[$method]} fit \
    --data.num_tasks 1 \
    --data.current_task 0 \
    --data.max_length 77 \
    --data.batch_size ${BATCH_SIZE} \
    --data.batch_size_zs ${BATCH_SIZE_ZS} \
    --data.num_workers ${NUM_WORKERS} \
    --data.config ${CONFIG_FILE} \
    --data.dataset_name ${dataset_name} \
    --data.root_dir ${ROOT_DIR} \
    --model.alpha 0.1 \
    --model.model_name ${MODEL_NAME} \
    --model.projection_dims 512 \
    --model.temperature 0.1 \
    --model.lr ${lr} \
    --model.lr_warmup_epochs 5 \
    --model.weight_decay 0.1 \
    --model.download_root ./ \
    --model.zero_shot_eval_interval ${ZERO_SHOT_EVAL_INTERVAL} \
    --trainer.accelerator npu \
    --trainer.precision 16 \
    --trainer.max_epochs ${MAX_EPOCHS} \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project ${PROJECT} \
    --trainer.logger.name ${dataset_name}-${MODEL_NAME}-${method}-lr-${lr} \
    --trainer.logger.log_model False \
    --trainer.strategy ddp_find_unused_parameters_true \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${dataset_name}-512/${method}-lr-${lr}
}

# 运行所有数据集
for DATASET_NAME in "${DATASETS[@]}"; do
  LR=${DATASET_LR_MAP[${DATASET_NAME}]}

  echo "Running training for dataset: ${DATASET_NAME} with lr: ${LR}"
  # 按不同的方法运行（比如 'vanilla'）
  run_training "fixedres ${DATASET_NAME} ${LR}

  echo "Completed training for dataset: ${DATASET_NAME}"
done

