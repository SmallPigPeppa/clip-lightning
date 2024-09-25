#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/ppio_net0/huggingface
# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表
DATASETS=("flickr30k" "coco2014" "wikiart" "patfig" "pet" "simpsons" "lexica" "styles" "kream" "sketch")
DATASETS=("coco2014")

# 定义检查点列表
CKPTS=(
  "ckpt/flickr30k-1024/distill_lora_v2-lr-8.6e-6-lr_text-1.3e-4.ckpt"
  "ckpt/coco2014-1024/distill_lora_v2-lr-5e-7-lr_text-4e-5.ckpt"
  "ckpt/pet-1024/distill_lora_v2-lr-1e-5-lr_text-4e-4.ckpt"
)


CKPTS_COMMA_JOINED=$(
  IFS=','
  echo "${CKPTS[*]}"
) # 用逗号拼接

# 其他参数
CONFIG_FILE=config.yaml
ROOT_DIR=/ppio_net0/torch_ds


# 遍历每个数据集
for DATASET_NAME in "${DATASETS[@]}"; do
  # 获取当前数据集的学习率
  LR=${DATASET_LR_MAP[${DATASET_NAME}]}

  echo "Running val for dataset: ${DATASET_NAME}"

  python cli_vanilla_zeroshot_hf.py validate \
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
    --model.projection_dims 512 \
    --model.temperature 0.1 \
    --model.lr 0. \
    --model.lr_warmup_epochs 5 \
    --model.weight_decay 0.1 \
    --model.download_root ./ \
    --model.recall_eval_interval 1 \
    --model.zero_shot_eval_interval 2 \
    --model.old_checkpoint_path ${CKPTS_COMMA_JOINED} \
    --trainer.accelerator gpu \
    --trainer.precision 16 \
    --trainer.max_epochs 1 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP-debug \
    --trainer.logger.name ${DATASET_NAME}-${MODEL_NAME}-lora-best \
    --trainer.logger.log_model False \
    --trainer.strategy ddp_find_unused_parameters_true \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${DATASET_NAME}-${MODEL_NAME}-lora-best

  echo "Completed training for dataset: ${DATASET_NAME}"
done

