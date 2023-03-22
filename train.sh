#!/bin/bash

python main.py \
-s veri \
-t veri \
-a linnet16 \
--root ../datasets \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--start-eval 0 \
--eval-freq 1 \
--print-freq 100 \
--random-erase \
--color-jitter \
--color-aug \
--pretrained-model linnet16-pretrained-imagenet1k-best.pth \
--save-dir logs/linnet16-veri
