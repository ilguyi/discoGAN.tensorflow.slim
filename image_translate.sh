#!/bin/bash

ROOT_DIR=$HOME/projects

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$ROOT_DIR/discoGAN.tensorflow.slim/exp1

BATCH_SIZE=$1

CUDA_VISIBLE_DEVICES=1 \
python image_translate.py \
    --checkpoint_dir=${TRAIN_DIR} \
    --is_all_checkpoints=True \
    --checkpoint_step=-1 \
    --batch_size=$BATCH_SIZE \
    --initial_learning_rate=0.0002 \
    --style_A='Male' \
    #--style_A='Blond_Hair' \
    #--style_B='Black_Hair' \
    #--constraint='Male' \
    #--constraint_type='1' \
    #--is_all_checkpoints=True \

