#MODEL_NAME=ViT-L/14
#MODEL_NAME=RN50
MODEL_NAME=ViT-B/16



#DATASET_NAME=fashion
#DATASET_NAME=patfig
#DATASET_NAME=pet
#DATASET_NAME=wikiart
#DATASET_NAME=midjourney
#DATASET_NAME=nouns
#DATASET_NAME=emoji
DATASET_NAME=face
#DATASET_NAME=pokemon

export HF_HOME=/ppio_net0/huggingface
python cli_vanilla_zeroshot_hf.py fit \
    --data.num_tasks 1 \
    --data.current_task 0 \
    --data.max_length 77 \
    --data.batch_size 128 \
    --data.batch_size_zs 32 \
    --data.num_workers 8 \
    --data.config config.yaml \
    --data.dataset_name ${DATASET_NAME} \
    --data.root_dir /ppio_net0/torch_ds \
    --model.model_name ${MODEL_NAME} \
    --model.projection_dims 1024 \
    --model.temperature 0.1 \
    --model.lr 1e-5 \
    --model.lr_warmup_epochs 5 \
    --model.weight_decay 0.1 \
    --model.download_root ./ \
    --model.zero_shot_eval_interval 40 \
    --trainer.accelerator gpu \
    --trainer.precision 16 \
    --trainer.max_epochs 40 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP-hf \
    --trainer.logger.name ${DATASET_NAME}-${MODEL_NAME}-vanilla \
    --trainer.logger.log_model False \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename  ${DATASET_NAME}-${MODEL_NAME}-vanilla