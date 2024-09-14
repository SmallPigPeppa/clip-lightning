#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/home/ma-user/work/wenzhuoliu/huggingface

# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表
#DATASETS=("emoji" "wikiart" "patfig" "fashion" "pet" "nouns" "shahnegar" "peanuts" "artbench" "hausavg" "simpsons" "lexica" "styles" "plans" "tomato" "kream" "sketch")
#DATASETS=("flickr30k" "coco2014" "wikiart" "patfig" "pet" "artbench" "simpsons" "lexica" "styles"  "kream" "sketch")
DATASETS=("simpsons" "lexica" "styles"  "kream" "sketch")
# 其他参数
CONFIG_FILE=config.yaml
ROOT_DIR=/home/ma-user/work/wenzhuoliu/torch_ds

# 遍历每个数据集
for DATASET_NAME in "${DATASETS[@]}"; do
    echo "Running training for dataset: ${DATASET_NAME}"

    python cli_vanilla_zeroshot_hf.py fit \
        --data.num_tasks 1 \
        --data.current_task 0 \
        --data.max_length 77 \
        --data.batch_size 128 \
        --data.batch_size_zs 32 \
        --data.num_workers 8 \
        --data.config ${CONFIG_FILE} \
        --data.dataset_name ${DATASET_NAME} \
        --data.root_dir ${ROOT_DIR} \
        --model.model_name ${MODEL_NAME} \
        --model.projection_dims 1024 \
        --model.temperature 0.1 \
        --model.lr 1e-5 \
        --model.lr_warmup_epochs 5 \
        --model.weight_decay 0.1 \
        --model.download_root ./ \
        --model.zero_shot_eval_interval 40 \
        --trainer.accelerator npu \
        --trainer.precision 16 \
        --trainer.max_epochs 40 \
        --trainer.log_every_n_steps 1 \
        --trainer.logger WandbLogger \
        --trainer.logger.project CLIP-hf-yd \
        --trainer.logger.name ${DATASET_NAME}-${MODEL_NAME}-vanilla \
        --trainer.logger.log_model False \
        --lr_monitor.logging_interval epoch \
        --model_checkpoint.dirpath ckpt \
        --model_checkpoint.save_weights_only True \
        --model_checkpoint.filename ${DATASET_NAME}-${MODEL_NAME}-vanilla

    echo "Completed training for dataset: ${DATASET_NAME}"
done
