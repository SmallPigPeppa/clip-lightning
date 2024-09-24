#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/ppio_net0/huggingface
ROOT_DIR=/ppio_net0/torch_ds
PROJECT=CLIP-1step-1024

# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表
DATASETS=("flickr30k" "coco2014" "wikiart" "patfig" "pet" "simpsons" "lexica" "styles" "kream" "sketch")
#DATASETS=("coco2014")
DATASETS=("flickr30k")

# 数据集学习率映射
declare -A DATASET_LR_MAP=(
  ["flickr30k"]=8e-6
  ["coco2014"]=1e-5
  ["wikiart"]=1e-5
  ["patfig"]=1e-5
  ["pet"]=1e-5
  ["simpsons"]=1e-5
  ["lexica"]=1e-5
  ["styles"]=1e-5
  ["kream"]=1e-5
  ["sketch"]=1e-5
  ["emoji"]=1e-5
  ["fashion"]=1e-5
  ["nouns"]=1e-5
  ["shahnegar"]=1e-5
  ["artbench"]=1e-5
  ["hausavg"]=1e-5
  ["food"]=1e-5
  ["clothes"]=1e-5
)

declare -A DATASET_LR_TEXT_MAP=(
  ["flickr30k"]=1.2e-4
  ["coco2014"]=2e-4
  ["wikiart"]=2e-4
  ["patfig"]=2e-4
  ["pet"]=2e-4
  ["simpsons"]=2e-4
  ["lexica"]=2e-4
  ["styles"]=2e-4
  ["kream"]=2e-4
  ["sketch"]=2e-4
  ["emoji"]=2e-4
  ["fashion"]=2e-4
  ["nouns"]=2e-4
  ["shahnegar"]=2e-4
  ["artbench"]=2e-4
  ["hausavg"]=2e-4
  ["food"]=2e-4
  ["clothes"]=2e-4
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
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${dataset_name}-1024/${method}-lr-${lr}-lr_text-${lr_text}
}

# 运行所有数据集
for DATASET_NAME in "${DATASETS[@]}"; do
  LR=${DATASET_LR_MAP[${DATASET_NAME}]}
  LR_TEXT=${DATASET_LR_TEXT_MAP[${DATASET_NAME}]}

  echo "Running training for dataset: ${DATASET_NAME} with lr: ${LR}"

  # 按不同的方法运行（比如 'distill_lora', 'vanilla'）
#  run_training "vanilla" ${DATASET_NAME} ${LR} ${LR_TEXT}
#  run_training "distill" ${DATASET_NAME} ${LR} ${LR_TEXT}
#  run_training "lora_v2" ${DATASET_NAME} ${LR} ${LR_TEXT}
#  run_training "lora" ${DATASET_NAME} ${LR} ${LR_TEXT}
#  run_training "distill_lora" ${DATASET_NAME} ${LR} ${LR_TEXT}
  run_training "distill_lora_v2" ${DATASET_NAME} ${LR} ${LR_TEXT}


  echo "Completed training for dataset: ${DATASET_NAME}"
done

#/ppio_net0/code/openapi.sh stop 46e358198c65fd38

