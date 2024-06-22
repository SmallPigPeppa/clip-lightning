python image_retrieval/cli.py fit \
    --data.dataset_name flickr30k \
    --data.artifact_id wandb/clip.lightning-image_retrieval/flickr-30k:latest \
    --data.train_batch_size 128 \
    --data.val_batch_size 128 \
    --data.max_length 200 \
    --model.image_encoder_alias resnet50 \
    --model.text_encoder_alias distilbert-base-uncased \
    --model.image_embedding_dims 2048 \
    --model.text_embedding_dims 768 \
    --model.projection_dims 256 \
    --trainer.precision 16 \
    --trainer.accelerator gpu \
    --trainer.max_epochs 20 \
    --trainer.log_every_n_steps 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.project CLIP \
    --trainer.logger.log_model all