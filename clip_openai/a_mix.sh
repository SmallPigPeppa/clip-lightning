#!/bin/bash


CKPTS=(
  "ckpt/flickr30k-1024/distill_lora_v2-lr-8.6e-6-lr_text-1.3e-4.ckpt"
  "ckpt/coco2014-1024/distill_lora_v2-lr-5e-7-lr_text-4e-5.ckpt"
  "ckpt/pet-1024/distill-lr-3e-5-lr_text-3e-5.ckpt"
#  "ckpt/lexica-1024/distill-lr-1e-4-lr_text-1e-4.ckpt"
#  "ckpt/simpsons-1024/distill-lr-5e-5-lr_text-5e-5.ckpt"
##  "ckpt/patfig-1024/distill_lora_v2-lr-5e-4-lr_text-5e-4.ckpt"
#  "ckpt/wikiart-1024/distill-lr-5e-5-lr_text-5e-5.ckpt"
)


# task1 coco2014
CHECKPOINT_PATH1="ckpt/flickr30k-1024/distill_lora_v2-lr-8.6e-6-lr_text-1.3e-4.ckpt"
CHECKPOINT_PATH2="ckpt/coco2014-1024/distill_lora_v2-lr-5e-7-lr_text-4e-5.ckpt"
OUTPUT_PATH="ckpt-cl/coco2014-task2.ckpt"
RATIO1=1
RATIO2=1

python a_mix.py "$CHECKPOINT_PATH1" "$CHECKPOINT_PATH2" "$OUTPUT_PATH" "$RATIO1" "$RATIO2"
