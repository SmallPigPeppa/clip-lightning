python image_retrieval/cli.py fit \
    --data.dataset_name flickr30k \
    --data.artifact_id wandb/clip.lightning-image_retrieval/flickr-30k:latest \
    --data.train_batch_size 256 \
    --data.val_batch_size 256 \
    --model.image_encoder_alias resnet50 \
    --model.text_encoder_alias distilbert-base-uncased \
    --model.image_embedding_dims 2048 \
    --model.image_encoder_lr 16e-4 \
    --model.image_encoder_pretrained true \
    --model.image_encoder_trainable true \
    --model.text_embedding_dims 768 \
    --model.text_encoder_lr 16e-5 \
    --model.text_encoder_trainable true \
    --model.projection_dims 512 \
    --model.head_lr 16e-3 \
    --trainer.accelerator npu \
    --trainer.max_epochs 40 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP \
    --trainer.logger.log_model all \
    --lr_monitor.logging_interval step