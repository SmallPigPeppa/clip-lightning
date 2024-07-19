# grad clip
# --trainer.gradient_clip_val 0.5 \
python image_retrieval/cli_aug_simclr.py fit \
    --data.dataset_name flickr30k_aug \
    --data.artifact_id wandb/clip.lightning-image_retrieval/flickr-30k:latest \
    --data.train_batch_size 256 \
    --data.val_batch_size 16 \
    --data.config config.yaml \
    --model.image_encoder_alias resnet50 \
    --model.text_encoder_alias distilbert-base-uncased \
    --model.image_encoder_pretrained True \
    --model.image_encoder_trainable True \
    --model.image_embedding_dims 2048 \
    --model.image_encoder_lr 1e-3 \
    --model.head_lr 1e-3 \
    --model.text_encoder_trainable True \
    --model.text_embedding_dims 768 \
    --model.text_encoder_lr 1e-5 \
    --model.projection_dims 512 \
    --trainer.accelerator npu \
    --trainer.precision 16 \
    --trainer.max_epochs 40 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP-PL \
    --trainer.logger.name r50-aug-bs256-debug-ILR-PLR-simclr \
    --trainer.logger.log_model False \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True
