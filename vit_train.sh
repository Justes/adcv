#!/bin/bash

python vit_main.py \
-s veri \
-t veri \
--root ../datasets \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.008 \
--weight-decay 0.0001 \
--max-epoch 1 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 64 \
--start-eval 0 \
--eval-freq 1 \
--print-freq 100 \
--pretrained-model ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
--save-dir logs/vit-base