#!/bin/bash

python main.py \
-s veri \
-t veri \
-a alexnet \
--root ../datasets \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 1 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--start-eval 0 \
--eval-freq 1 \
--print-freq 100 \
--random-erase \
--color-jitter \
--color-aug \
--pretrained-model ../pretrained/alexnet-7be5be79.pth \
--save-dir logs/alexnet-veri
