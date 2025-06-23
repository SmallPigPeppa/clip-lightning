#!/usr/bin/env bash

# --------------------
# 环境 & 路径设置
# --------------------
export HF_HOME=/mnt/bn/liuwenzhuo-hl-data/hf_cache
export WANDB_BASE_URL=https://api.bandw.top

ROOT_DIR=/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets
MODEL_DIR=/mnt/bn/liuwenzhuo-hl-data/openclip_cache
CKPT_DIR=/mnt/bn/liuwenzhuo-hl-data/ckpt/clip_msun
PROJECT=CLIP-MSUN
MODEL_NAME=RN50
METHOD=msun



# --------------------
# 通用训练参数
# --------------------
CONFIG_FILE=config.yaml
ZERO_SHOT_EVAL_INTERVAL=40
MAX_EPOCHS=80
BATCH_SIZE=512
BATCH_SIZE_ZS=32
NUM_WORKERS=8
NUM_GPUS=1
TOTAL_BATCH_SIZE=$(( BATCH_SIZE * NUM_GPUS ))

dataset="flickr30k"
#dataset="coco2014"
lr=1e-5

# --------------------
# 训练函数
# --------------------

python3 cli_val_anyres_msun_block4.py validate \
  --ckpt_path ${CKPT_DIR}/${METHOD}-${dataset}-${MODEL_NAME}-lr${lr}-bs${TOTAL_BATCH_SIZE}-v3.ckpt \
  --data.num_tasks 1 \
  --data.current_task 0 \
  --data.max_length 77 \
  --data.batch_size ${BATCH_SIZE} \
  --data.batch_size_zs ${BATCH_SIZE_ZS} \
  --data.num_workers ${NUM_WORKERS} \
  --data.config ${CONFIG_FILE} \
  --data.dataset ${dataset} \
  --data.root_dir ${ROOT_DIR} \
  --model.model_name ${MODEL_NAME} \
  --model.projection_dims 512 \
  --model.download_root ${MODEL_DIR} \
  --model.zero_shot_eval_interval ${ZERO_SHOT_EVAL_INTERVAL} \
  --trainer.accelerator npu \
  --trainer.devices ${NUM_GPUS} \
  --trainer.precision 16 \
  --trainer.max_epochs ${MAX_EPOCHS} \
  --trainer.log_every_n_steps 1 \
  --trainer.logger WandbLogger \
  --trainer.logger.project ${PROJECT} \
  --trainer.logger.name val-${METHOD}-${dataset}-${MODEL_NAME}-lr${lr}-bs${TOTAL_BATCH_SIZE}-block4 \
  --trainer.logger.log_model False \
  --trainer.logger.offline False \
  --trainer.strategy ddp_find_unused_parameters_true \
  --lr_monitor.logging_interval epoch \
  --model_checkpoint.dirpath ${CKPT_DIR} \
  --model_checkpoint.save_weights_only True \
  --model_checkpoint.filename ${METHOD}-${dataset}-${MODEL_NAME}-lr${lr}-bs${TOTAL_BATCH_SIZE}-block4
