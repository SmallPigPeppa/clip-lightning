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


## task2 coco2014
#CHECKPOINT_PATH1="ckpt/flickr30k-1024/distill_lora_v2-lr-8.6e-6-lr_text-1.3e-4.ckpt"
#CHECKPOINT_PATH2="ckpt/coco2014-1024/distill_lora_v2-lr-5e-7-lr_text-4e-5.ckpt"
#OUTPUT_PATH="ckpt-cl/task2-coco2014.ckpt"
#RATIO1=1
#RATIO2=1
#
#python a_mix.py "$CHECKPOINT_PATH1" "$CHECKPOINT_PATH2" "$OUTPUT_PATH" "$RATIO1" "$RATIO2"
#
#
#
## task3 pet
#CHECKPOINT_PATH1="ckpt-cl/task2-coco2014.ckpt"
#CHECKPOINT_PATH2="ckpt/pet-1024/distill-lr-3e-5-lr_text-3e-5.ckpt"
#OUTPUT_PATH="ckpt-cl/task3-pet.ckpt"
#RATIO1=2
#RATIO2=1
#
#python a_mix.py "$CHECKPOINT_PATH1" "$CHECKPOINT_PATH2" "$OUTPUT_PATH" "$RATIO1" "$RATIO2"




## task4 lexical
#CHECKPOINT_PATH1="ckpt-cl/task3-pet.ckpt"
#CHECKPOINT_PATH2="ckpt/lexica-1024/distill-lr-5e-5-lr_text-5e-5.ckpt"
#OUTPUT_PATH="ckpt-cl/task4-lexica.ckpt"
#RATIO1=2
#RATIO2=1
#
#python a_mix.py "$CHECKPOINT_PATH1" "$CHECKPOINT_PATH2" "$OUTPUT_PATH" "$RATIO1" "$RATIO2"


#
##
## task5 simpsons
#CHECKPOINT_PATH1="ckpt-cl/task4-lexica.ckpt"
#CHECKPOINT_PATH2="ckpt/simpsons-1024/distill-lr-5e-5-lr_text-5e-5-v2.ckpt"
#OUTPUT_PATH="ckpt-cl/task5-simpsons.ckpt"
#RATIO1=2
#RATIO2=1
#
#python a_mix.py "$CHECKPOINT_PATH1" "$CHECKPOINT_PATH2" "$OUTPUT_PATH" "$RATIO1" "$RATIO2"
#
#
## task 6 wikiart
#CHECKPOINT_PATH1="ckpt-cl/task5-simpsons.ckpt"
#CHECKPOINT_PATH2="ckpt/wikiart-1024/distill-lr-5e-5-lr_text-5e-5.ckpt"
#OUTPUT_PATH="ckpt-cl/task6-wikiart.ckpt"
#RATIO1=2
#RATIO2=1
#
#python a_mix.py "$CHECKPOINT_PATH1" "$CHECKPOINT_PATH2" "$OUTPUT_PATH" "$RATIO1" "$RATIO2"




# task 7 kream
CHECKPOINT_PATH1="ckpt-cl/task6-wikiart.ckpt"
CHECKPOINT_PATH2="ckpt/kream-1024/distill_lora_v2-lr-5e-5-lr_text-5e-5.ckpt"
OUTPUT_PATH="ckpt-cl/task7-kream-lora.ckpt"
RATIO1=2
RATIO2=1

python a_mix.py "$CHECKPOINT_PATH1" "$CHECKPOINT_PATH2" "$OUTPUT_PATH" "$RATIO1" "$RATIO2"


#
## task6 patfig
#CHECKPOINT_PATH1="ckpt-cl/task5-simpsons.ckpt"
#CHECKPOINT_PATH2="ckpt/patfig-1024/distill-lr-3e-5-lr_text-3e-5.ckpt"
#OUTPUT_PATH="ckpt-cl/task6-patfig.ckpt"
#RATIO1=2
#RATIO2=1
#
#python a_mix.py "$CHECKPOINT_PATH1" "$CHECKPOINT_PATH2" "$OUTPUT_PATH" "$RATIO1" "$RATIO2"
