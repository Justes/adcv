#!/bin/bash

python vit_reid.py \
-s veri \
-t veri \
--root ../datasets \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 1 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 64 \
--start-eval 0 \
--eval-freq 1 \
--print-freq 100 \
--pretrained-model ../pretrained/vit_base_patch16_224_in21k-e5005f0a.pth \
--save-dir logs/linnet16-veri