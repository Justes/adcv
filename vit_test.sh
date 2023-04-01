#!/bin/bash

python vit_reid.py \
-s veri \
-t veri \
--root ../datasets \
--height 224 \
--width 224 \
--test-batch-size 100 \
--evaluate \
--print-freq 100 \
--load-weights ../vit_base_veri.pth \
#--load-weights ../2023-04-01_model.pth.tar-60 \
--save-dir logs/vit
