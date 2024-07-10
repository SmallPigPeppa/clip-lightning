# grad clip
# --trainer.gradient_clip_val 0.5 \
python image_retrieval/cli_aug.py fit \
    --data.dataset_name flickr30k_aug \
    --data.artifact_id wandb/clip.lightning-image_retrieval/flickr-30k:latest \
    --data.train_batch_size 32 \
    --data.val_batch_size 32 \
    --data.config config.yaml \
    --model.image_encoder_alias resnet50 \
    --model.text_encoder_alias distilbert-base-uncased \
    --model.image_encoder_pretrained True \
    --model.image_encoder_trainable True \
    --model.image_embedding_dims 2048 \
    --model.image_encoder_lr 1e-4 \
    --model.text_encoder_trainable True \
    --model.text_embedding_dims 768 \
    --model.text_encoder_lr 1e-5 \
    --model.projection_dims 512 \
    --model.head_lr 1e-3 \
    --trainer.accelerator npu \
    --trainer.max_epochs 40 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP-lighting \
    --trainer.logger.name r50-aug-bs32 \
    --trainer.logger.log_model False \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True