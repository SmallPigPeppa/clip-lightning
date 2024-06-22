python image_retrieval/cli.py fit \
    --data.dataset_name flickr8k \
    --data.artifact_id wandb/clip.lightning-image_retrieval/flickr-8k:latest \
    --data.train_batch_size 128 \
    --data.val_batch_size 128 \
    --model.image_encoder_alias resnet50 \
    --model.text_encoder_alias distilbert-base-uncased \
    --trainer.accelerator gpu \
    --trainer.max_epochs 20 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP \
    --trainer.logger.log_model all