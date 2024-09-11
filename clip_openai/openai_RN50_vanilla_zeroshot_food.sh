#MODEL_NAME=ViT-L/14
#MODEL_NAME=RN50
MODEL_NAME=ViT-B/16
python cli_vanilla_zeroshot.py fit \
    --data.num_tasks 1 \
    --data.current_task 0 \
    --data.max_length 77 \
    --data.batch_size 128 \
    --data.batch_size_zs 32 \
    --data.num_workers 8 \
    --data.config config.yaml \
    --data.dataset_name food \
    --data.root_dir /ppio_net0/torch_ds \
    --model.model_name ${MODEL_NAME} \
    --model.projection_dims 1024 \
    --model.temperature 0.1 \
    --model.lr 1e-5 \
    --model.lr_warmup_epochs 5 \
    --model.weight_decay 0. \
    --model.download_root ./ \
    --trainer.accelerator gpu \
    --trainer.precision 16 \
    --trainer.max_epochs 40 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP-food \
    --trainer.logger.name ${MODEL_NAME}-vanilla \
    --trainer.logger.log_model False \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename  RN50-vanilla