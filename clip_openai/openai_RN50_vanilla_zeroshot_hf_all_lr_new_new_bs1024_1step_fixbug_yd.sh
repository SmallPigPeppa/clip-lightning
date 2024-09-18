#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/home/ma-user/work/wenzhuoliu/huggingface


# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表
DATASETS=("flickr30k" "coco2014" "wikiart" "patfig" "pet" "simpsons" "lexica" "styles" "kream" "sketch")
DATASETS=("coco2014" "wikiart" "patfig" "pet" "simpsons" "lexica" "styles" "kream" "sketch")
#DATASETS=("flickr30k")
DATASETS=("coco2014")
DATASETS=("wikiart")


# 其他参数
CONFIG_FILE=config.yaml
ROOT_DIR=/home/ma-user/work/wenzhuoliu/torch_ds

# 定义不同数据集的学习率
declare -A DATASET_LR_MAP=(
  ["flickr30k"]=1e-5
  ["coco2014"]=2e-5
  ["wikiart"]=1e-5
  ["patfig"]=1e-5
  ["pet"]=1e-5
  ["simpsons"]=1e-5
  ["lexica"]=1e-5
  ["styles"]=1e-5
  ["kream"]=1e-5
  ["sketch"]=1e-5
)

# 遍历每个数据集
for DATASET_NAME in "${DATASETS[@]}"; do
  # 获取当前数据集的学习率
  LR=${DATASET_LR_MAP[${DATASET_NAME}]}

  echo "Running training for dataset: ${DATASET_NAME} with lr: ${LR}"



#  python cli_distill_zeroshot_lora_best.py fit \
#    --data.num_tasks 1 \
#    --data.current_task 0 \
#    --data.max_length 77 \
#    --data.batch_size 128 \
#    --data.batch_size_zs 32 \
#    --data.num_workers 8 \
#    --data.config ${CONFIG_FILE} \
#    --data.dataset_name ${DATASET_NAME} \
#    --data.root_dir ${ROOT_DIR} \
#    --model.model_name ${MODEL_NAME} \
#    --model.projection_dims 512 \
#    --model.temperature 0.1 \
#    --model.lr ${LR} \
#    --model.lr_warmup_epochs 5 \
#    --model.weight_decay 0.1 \
#    --model.download_root ./ \
#    --model.zero_shot_eval_interval 40 \
#    --trainer.accelerator npu \
#    --trainer.precision 16 \
#    --trainer.max_epochs 40 \
#    --trainer.log_every_n_steps 1 \
#    --trainer.logger WandbLogger \
#    --trainer.logger.project CLIP-1step-1024-yd \
#    --trainer.logger.name ${DATASET_NAME}-${MODEL_NAME}-dislora \
#    --trainer.logger.log_model False \
#    --trainer.strategy ddp_find_unused_parameters_true \
#    --lr_monitor.logging_interval epoch \
#    --model_checkpoint.dirpath ckpt \
#    --model_checkpoint.save_weights_only True \
#    --model_checkpoint.filename ${DATASET_NAME}-1024/dislora

  python cli_distill_zeroshot_lora_best_new.py fit \
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
    --model.projection_lr_scale 10 \
    --model.temperature 0.1 \
    --model.lr ${LR} \
    --model.lr_warmup_epochs 5 \
    --model.weight_decay 0.1 \
    --model.download_root ./ \
    --model.zero_shot_eval_interval 40 \
    --trainer.accelerator npu \
    --trainer.precision 16 \
    --trainer.max_epochs 40 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP-1step-1024-yd \
    --trainer.logger.name ${DATASET_NAME}-${MODEL_NAME}-dislora-new \
    --trainer.logger.log_model False \
    --trainer.strategy ddp_find_unused_parameters_true \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${DATASET_NAME}-1024/dislora-new
#
#  python cli_distill_zeroshot.py fit \
#    --data.num_tasks 1 \
#    --data.current_task 0 \
#    --data.max_length 77 \
#    --data.batch_size 128 \
#    --data.batch_size_zs 32 \
#    --data.num_workers 8 \
#    --data.config ${CONFIG_FILE} \
#    --data.dataset_name ${DATASET_NAME} \
#    --data.root_dir ${ROOT_DIR} \
#    --model.model_name ${MODEL_NAME} \
#    --model.projection_dims 512 \
#    --model.temperature 0.1 \
#    --model.lr ${LR} \
#    --model.lr_warmup_epochs 5 \
#    --model.weight_decay 0.1 \
#    --model.download_root ./ \
#    --model.zero_shot_eval_interval 40 \
#    --trainer.accelerator npu \
#    --trainer.precision 16 \
#    --trainer.max_epochs 40 \
#    --trainer.log_every_n_steps 1 \
#    --trainer.logger WandbLogger \
#    --trainer.logger.project CLIP-1step-1024-yd \
#    --trainer.logger.name ${DATASET_NAME}-${MODEL_NAME}-distill \
#    --trainer.logger.log_model False \
#    --trainer.strategy ddp_find_unused_parameters_true \
#    --lr_monitor.logging_interval epoch \
#    --model_checkpoint.dirpath ckpt \
#    --model_checkpoint.save_weights_only True \
#    --model_checkpoint.filename ${DATASET_NAME}-1024/distill


#  python cli_vanilla_zeroshot_hf.py fit \
#    --data.num_tasks 1 \
#    --data.current_task 0 \
#    --data.max_length 77 \
#    --data.batch_size 128 \
#    --data.batch_size_zs 32 \
#    --data.num_workers 8 \
#    --data.config ${CONFIG_FILE} \
#    --data.dataset_name ${DATASET_NAME} \
#    --data.root_dir ${ROOT_DIR} \
#    --model.model_name ${MODEL_NAME} \
#    --model.projection_dims 512 \
#    --model.temperature 0.1 \
#    --model.lr ${LR} \
#    --model.lr_warmup_epochs 5 \
#    --model.weight_decay 0.1 \
#    --model.download_root ./ \
#    --model.zero_shot_eval_interval 40 \
#    --trainer.accelerator npu \
#    --trainer.precision 16 \
#    --trainer.max_epochs 40 \
#    --trainer.log_every_n_steps 1 \
#    --trainer.logger WandbLogger \
#    --trainer.logger.project CLIP-1step-1024-yd \
#    --trainer.logger.name ${DATASET_NAME}-${MODEL_NAME}-vanilla \
#    --trainer.logger.log_model False \
#    --trainer.strategy ddp_find_unused_parameters_true \
#    --lr_monitor.logging_interval epoch \
#    --model_checkpoint.dirpath ckpt \
#    --model_checkpoint.save_weights_only True \
#    --model_checkpoint.filename ${DATASET_NAME}-1024/vanilla

  echo "Completed training for dataset: ${DATASET_NAME}"
done

/ppio_net0/code/openapi.sh stop 5107be1343913d61
