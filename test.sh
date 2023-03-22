#!/bin/bash

python main.py \
-s veri \
-t veri \
-a linnet16 \
--root ../datasets \
--height 224 \
--width 224 \
--test-batch-size 100 \
--evaluate \
--print-freq 100 \
--load-weights ../2023-03-22_best_model.pth.tar \
--save-dir logs/eval-veri
