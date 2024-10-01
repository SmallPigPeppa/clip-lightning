#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/ppio_net0/huggingface
ROOT_DIR=/ppio_net0/torch_ds
PROJECT=CLIP-1step-1024-params

# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表
DATASETS=("flickr30k")

# 数据集学习率映射
declare -A DATASET_LR_MAP=(
  ["flickr30k"]=8.6e-6
)

declare -A DATASET_LR_TEXT_MAP=(
  ["flickr30k"]=1.3e-4
)



# 脚本方法映射
declare -A METHOD_MAP=(
  ["vanilla"]="cli_vanilla_zeroshot_hf.py"
  ["lora"]="cli_vanilla_zeroshot_lora_best.py"
  ["lora_v2"]="cli_vanilla_zeroshot_lora_best_v2.py"
  ["distill"]="cli_distill_zeroshot.py"
  ["distill_v4"]="cli_distill_zeroshot_v4.py"
  ["distill_lora"]="cli_distill_zeroshot_lora_best.py"
  ["distill_lora_v2"]="cli_distill_zeroshot_lora_best_v2.py"
  ["distill_lora_v3"]="cli_distill_zeroshot_lora_best_v3.py"
  ["zscl"]="cli_distill_zeroshot_zscl.py"
  ["modx"]="cli_distill_zeroshot_modx.py"
  ["ewc"]="cli_distill_zeroshot_ewc_new.py"
)

# 其他参数
CONFIG_FILE=config.yaml
ZERO_SHOT_EVAL_INTERVAL=5
MAX_EPOCHS=40
BATCH_SIZE=64
BATCH_SIZE_ZS=32
NUM_WORKERS=8

# 函数：运行训练
run_training() {
  local method=$1
  local dataset_name=$2
  local lr=$3
  local lr_text=$4

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
    --model_checkpoint.dirpath ckpt-exp1 \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${dataset_name}-1024/${method}-lr-${lr}-lr_text-${lr_text}
}

# 运行所有数据集
for DATASET_NAME in "${DATASETS[@]}"; do
  LR=${DATASET_LR_MAP[${DATASET_NAME}]}
  LR_TEXT=${DATASET_LR_TEXT_MAP[${DATASET_NAME}]}

  echo "Running training for dataset: ${DATASET_NAME} with lr: ${LR}"

  # 按不同的方法运行（比如 'distill_lora', 'vanilla'）
  run_training "ewc" ${DATASET_NAME} ${LR} ${LR_TEXT}


  echo "Completed training for dataset: ${DATASET_NAME}"
done

#/ppio_net0/code/openapi.sh stop 14ee9a05e41fc7a4

