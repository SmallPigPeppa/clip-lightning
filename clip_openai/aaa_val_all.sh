#!/bin/bash

# 设置 Hugging Face home 目录
export HF_HOME=/ppio_net0/huggingface
# 模型名称
MODEL_NAME=ViT-B/16

# 数据集列表，拼接成逗号分隔的字符串
datasets=("flickr30k" "coco2014" "wikiart" "patfig" "pet" "simpsons" "lexica" "styles" "kream" "sketch")
datasets=(
"flickr30k"
"coco2014"
"pet"
#"lexica"
#"simpsons"
#"patfig"
#"wikiart"
#"styles"
#"kream"
#"sketch"
)
#datasets=("patfig" "wikiart" "styles" "kream" "sketch")
DATASETS=$(IFS=','; echo "${datasets[*]}")

# 定义检查点列表
CKPTS=(
  "ckpt/flickr30k-1024/distill_lora_v2-lr-8.6e-6-lr_text-1.3e-4.ckpt"
  "ckpt/coco2014-1024/distill_lora_v2-lr-5e-7-lr_text-4e-5.ckpt"
#  "ckpt/pet-1024/distill_lora_v2-lr-1e-5-lr_text-4e-4.ckpt"
"ckpt/pet-1024/distill-lr-2e-5-lr_text-2e-5.ckpt"
#  "ckpt/lexica-1024/distill_lora_v2-lr-3e-5-lr_text-3e-5.ckpt"
)

# 拼接检查点
CKPTS_COMMA_JOINED=$(IFS=','; echo "${CKPTS[*]}")

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
    --model.evaluate_zero_shot False \
    --model.result_path metrics_evaluation.xlsx \
    --model.model_name ${MODEL_NAME} \
    --model.download_root ./ \
    --model.old_checkpoint_path ${CKPTS_COMMA_JOINED} \
    --trainer.accelerator gpu \
    --trainer.precision 16 \
    --trainer.max_epochs 1 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP-debug \
    --trainer.logger.name ${MODEL_NAME}-eval-all \
    --trainer.logger.log_model False \
    --trainer.strategy ddp_find_unused_parameters_true \

#python cli_val.py validate \
#    --data.dataset_name ${DATASETS} \
#    --data.max_length 77 \
#    --data.batch_size 128 \
#    --data.batch_size_zs 32 \
#    --data.num_workers 8 \
#    --data.config ${CONFIG_FILE} \
#    --data.root_dir ${ROOT_DIR} \
#    --model.evaluate_zero_shot False \
#    --model.result_path metrics_evaluation-vanilla.xlsx \
#    --model.model_name ${MODEL_NAME} \
#    --model.download_root ./ \
#    --trainer.accelerator gpu \
#    --trainer.precision 16 \
#    --trainer.max_epochs 1 \
#    --trainer.log_every_n_steps 1 \
#    --trainer.logger WandbLogger \
#    --trainer.logger.project CLIP-debug \
#    --trainer.logger.name ${MODEL_NAME}-eval-all \
#    --trainer.logger.log_model False \
#    --trainer.strategy ddp_find_unused_parameters_true \



echo "Completed evaluation for all datasets"
