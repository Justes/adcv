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
--eval-freq 5 \
--print-freq 100 \
--no-pretrained \
--random-erase \
--color-jitter \
--color-aug \
--vertical-flip \
--rotation 20 \
--save-dir logs/linnet16-veri
