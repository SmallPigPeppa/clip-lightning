#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/ppio_net0/huggingface
# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表，拼接成逗号分隔的字符串
datasets=("flickr30k" "coco2014" "wikiart" "patfig" "pet" "simpsons" "lexica" "styles" "kream" "sketch")
datasets=(
  "flickr30k"
#  "coco2014"
#  "pet"
#  "lexica"
#  "simpsons"
#  "patfig"
#  "wikiart"
#  "styles"
#  "kream"
#  "sketch"
)
DATASETS=$(
  IFS=','
  echo "${datasets[*]}"
)


#CKPT="ckpt-cl/task6-wikiart.ckpt"
#CKPT="ckpt-cl/task7-kream-lora.ckpt"
CKPT="ckpt-cl/task7-kream.ckpt"
CKPT="ckpt-cl/task7-kream-5e-5.ckpt"
CKPT="ckpt-cl/task8-sketch-5e-5.ckpt"


# 其他参数
CONFIG_FILE=config.yaml
ROOT_DIR=/ppio_net0/torch_ds

# 评估所有数据集
python cli_val.py validate \
  --data.dataset_name ${DATASETS} \
  --data.max_length 77 \
  --data.batch_size 128 \
  --data.batch_size_zs 32 \
  --data.num_workers 8 \
  --data.config ${CONFIG_FILE} \
  --data.root_dir ${ROOT_DIR} \
  --model.evaluate_zero_shot True \
  --model.result_path metrics_evaluation.xlsx \
  --model.model_name ${MODEL_NAME} \
  --model.download_root ./ \
  --model.old_checkpoint_path ${CKPT} \
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
