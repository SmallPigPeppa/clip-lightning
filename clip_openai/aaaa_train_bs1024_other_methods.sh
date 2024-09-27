#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/ppio_net0/huggingface
ROOT_DIR=/ppio_net0/torch_ds
PROJECT=CLIP-1step-1024

# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表
DATASETS=("flickr30k" "coco2014" "pet" "lexica" "simpsons")

LR=4e-5

# 脚本方法映射
declare -A METHOD_MAP=(
  ["fine-tune"]="cli_vanilla_zeroshot_hf.py"
  ["zscl"]="cli_distill_zeroshot_zscl.py"
)

# 其他参数
CONFIG_FILE=config.yaml
ZERO_SHOT_EVAL_INTERVAL=40
MAX_EPOCHS=40
BATCH_SIZE=128
BATCH_SIZE_ZS=32
NUM_WORKERS=8

# 初始化变量以存储上一个数据集的ckpt
PREV_CKPT=None

# 函数：运行训练
run_training() {
  local method=$1
  local dataset_name=$2
  local lr=$3
  local lr_text=$4
  local old_ckpt_path=$5

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
    --model.model_name ${MODEL_NAME} \
    --model.projection_dims 512 \
    --model.temperature 0.1 \
    --model.lr ${lr} \
    --model.lr_text ${lr_text} \
    --model.lr_warmup_epochs 5 \
    --model.weight_decay 0.1 \
    --model.download_root ./ \
    --model.zero_shot_eval_interval ${ZERO_SHOT_EVAL_INTERVAL} \
    --model.old_checkpoint_path ${old_ckpt_path} \
    --trainer.accelerator gpu \
    --trainer.precision 16 \
    --trainer.max_epochs ${MAX_EPOCHS} \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project ${PROJECT} \
    --trainer.logger.name ${dataset_name}-${method}-lr-${lr}-lr_text-${lr_text} \
    --trainer.logger.log_model False \
    --trainer.strategy ddp_find_unused_parameters_true \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt-others \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${dataset_name}-1024/${method}-lr-${lr}-lr_text-${lr_text}
}

# 运行所有数据集
for i in "${!DATASETS[@]}"; do
  DATASET_NAME=${DATASETS[$i]}

  echo "Running training for dataset: ${DATASET_NAME} with lr: ${LR}"

  # 设置ckpt路径，如果是第一个数据集，ckpt为None
  if [[ $i -eq 0 ]]; then
    CKPT=None
  else
    CKPT="ckpt-others/${DATASETS[$((i-1))]}-lr-${LR}-lr_text-${LR}.ckpt"
  fi

  # 按不同的方法运行
  run_training "fine-tune" ${DATASET_NAME} ${LR} ${LR} ${CKPT}
  run_training "zscl" ${DATASET_NAME} ${LR} ${LR} ${CKPT}

  # 更新 PREV_CKPT 为当前数据集的 checkpoint
  PREV_CKPT="ckpt-others/${DATASET_NAME}-lr-${LR}-lr_text-${LR}.ckpt"

  echo "Completed training for dataset: ${DATASET_NAME}"
done

/ppio_net0/code/openapi.sh stop a2b028e85b48907c
