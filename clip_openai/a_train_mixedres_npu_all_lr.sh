#!/usr/bin/env bash

# --------------------
# 环境 & 路径设置
# --------------------
export HF_HOME=/mnt/bn/liuwenzhuo-hl-data/hf_cache
export WANDB_BASE_URL=https://api.bandw.top

ROOT_DIR=/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets
MODEL_DIR=/mnt/bn/liuwenzhuo-hl-data/openclip_cache
CKPT_DIR=/mnt/bn/liuwenzhuo-hl-data/ckpt/clip_msun/mixedres
PROJECT=CLIP-MSUN
MODEL_NAME=RN50

# --------------------
# 数据集列表
# --------------------
DATASETS=(flickr30k coco2014)

# --------------------
# 学习率列表
# --------------------
#LR_LIST=(5e-6 1e-5)
LR_LIST=(1.25e-6)

# --------------------
# 通用训练参数
# --------------------
CONFIG_FILE=config.yaml
ZERO_SHOT_EVAL_INTERVAL=40
MAX_EPOCHS=80
BATCH_SIZE=512
BATCH_SIZE_ZS=32
NUM_WORKERS=8
NUM_GPUS=8
TOTAL_BATCH_SIZE=$(( BATCH_SIZE * NUM_GPUS ))


# --------------------
# 训练函数
# --------------------
run_training() {
  local dataset=$1
  local lr=$2

  python3 cli_mixedres.py fit \
    --data.num_tasks 1 \
    --data.current_task 0 \
    --data.max_length 77 \
    --data.batch_size ${BATCH_SIZE} \
    --data.batch_size_zs ${BATCH_SIZE_ZS} \
    --data.num_workers ${NUM_WORKERS} \
    --data.config ${CONFIG_FILE} \
    --data.dataset_name ${dataset} \
    --data.root_dir ${ROOT_DIR} \
    --model.model_name ${MODEL_NAME} \
    --model.projection_dims 512 \
    --model.temperature 0.1 \
    --model.lr_visual ${lr} \
    --model.lr_text   ${lr} \
    --model.lr_warmup_epochs 5 \
    --model.weight_decay 0.1 \
    --model.download_root ${MODEL_DIR} \
    --model.zero_shot_eval_interval ${ZERO_SHOT_EVAL_INTERVAL} \
    --trainer.accelerator npu \
    --trainer.devices ${NUM_GPUS} \
    --trainer.precision 16 \
    --trainer.max_epochs ${MAX_EPOCHS} \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project ${PROJECT} \
    --trainer.logger.name ${dataset}-${MODEL_NAME}-lr${lr}-bs${TOTAL_BATCH_SIZE} \
    --trainer.logger.log_model False \
    --trainer.logger.offline False \
    --trainer.strategy ddp_find_unused_parameters_true \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ${CKPT_DIR} \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${MODEL_NAME}-{dataset}-lr${lr}-bs${TOTAL_BATCH_SIZE}
}

# --------------------
# 嵌套循环：按学习率->数据集 运行
# --------------------
for lr in "${LR_LIST[@]}"; do
  for ds in "${DATASETS[@]}"; do
    echo -e "\n=== Training on ${ds} with lr=${lr} ==="
    run_training "${ds}" "${lr}"
    echo "=== Completed ${ds}, lr=${lr} ==="
  done
done
