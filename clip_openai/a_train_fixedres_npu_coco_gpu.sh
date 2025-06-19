#!/usr/bin/env


# 设置 Hugging Face home 目录
export HF_HOME=/mnt/hdfs/byte_content_security/user/liuwenzhuo/hf_cache
export WANDB_BASE_URL=https://api.bandw.top
ROOT_DIR=/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets
MODEL_DIR=/mnt/hdfs/byte_content_security/user/liuwenzhuo/openclip_cache
CKPT_DIR=/mnt/hdfs/byte_content_security/user/liuwenzhuo/ckpt/clip_msun
PROJECT=CLIP-MSUN



# 模型名称
MODEL_NAME=RN101

# 数据集列表（只修改这里可灵活增删）
DATASETS=("flickr30k" "coco2014")
#DATASETS=("flickr30k")

# 为每个数据集单独定义视觉分支和文本分支的学习率
declare -A DATASET_LR_VISUAL_MAP=(
  ["flickr30k"]=5e-5
  ["coco2014"]=5e-5
)
declare -A DATASET_LR_TEXT_MAP=(
  ["flickr30k"]=5e-5
  ["coco2014"]=5e-5
)

# 脚本方法映射
declare -A METHOD_MAP=(
  ["fixedres"]="cli_fixedres.py"
)

# 其他公共参数
CONFIG_FILE=config.yaml
ZERO_SHOT_EVAL_INTERVAL=40
MAX_EPOCHS=80
BATCH_SIZE=128
BATCH_SIZE_ZS=32
NUM_WORKERS=8
NUM_GPUS=8
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# 函数：运行训练，接收视觉和文本学习率两个参数
run_training() {
  local method=$1
  local dataset_name=$2
  local lr_visual=$3
  local lr_text=$4

  python3 ${METHOD_MAP[$method]} fit \
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
    --model.lr_visual ${lr_visual} \
    --model.lr_text ${lr_text} \
    --model.lr_warmup_epochs 5 \
    --model.weight_decay 0.1 \
    --model.download_root ${MODEL_DIR} \
    --model.zero_shot_eval_interval ${ZERO_SHOT_EVAL_INTERVAL} \
    --trainer.accelerator gpu \
    --trainer.devices ${NUM_GPUS} \
    --trainer.precision 16 \
    --trainer.max_epochs ${MAX_EPOCHS} \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project ${PROJECT} \
    --trainer.logger.name ${dataset_name}-${MODEL_NAME}-${method}-vis${lr_visual}-txt${lr_text}-bs${TOTAL_BATCH_SIZE} \
    --trainer.logger.log_model False \
    --trainer.logger.offline False \
    --trainer.strategy ddp_find_unused_parameters_true \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ${CKPT_DIR} \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${dataset_name}-${method}-vis${lr_visual}-txt${lr_text}-bs${TOTAL_BATCH_SIZE}
}

# 循环运行所有数据集
for DATASET_NAME in "${DATASETS[@]}"; do
  lr_vis=${DATASET_LR_VISUAL_MAP[$DATASET_NAME]}
  lr_txt=${DATASET_LR_TEXT_MAP[$DATASET_NAME]}

  echo "Running training for dataset: ${DATASET_NAME} with lr_visual=${lr_vis}, lr_text=${lr_txt}"
  run_training "fixedres" "${DATASET_NAME}" "${lr_vis}" "${lr_txt}"
  echo "Completed training for dataset: ${DATASET_NAME}"
done
