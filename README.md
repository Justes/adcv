# Advanced Computer Vision


# entry main.py

# Default parameters

src/args.py
start-epoch = 0
max-epoch = 60
start-eval = 0 (start evaluate performance after this parameter)
print-freq = 10 log print freqency, 10 batch print one log

dataset_kwargs function in args.py add params

# Loss
xent: cross entropy loss
htri: triple loss (Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737)


#Linnet-16

base first conv 7*7, s=2, no init, use bn
v1 first conv 7*7, s=2, no init, remove bn use bias
v2 first conv 7*7, s=1, maxpool, no init, use bn
v3 first conv 7*7, s=1, maxpool, use bn and conv use kaiming_normal

# Data Augmentation

utils/transforms.py

# Train 
!python main.py \
-s veri \
-t veri \
-a resnet50 \
--root /content \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--start-eval 20 \
--eval-freq 5 \
--print-freq 100 \
--save-dir logs/resnet50-veri


# Test
!python main.py \
-s veri \
-t veri \
-a resnet50 \
--root /content \
--height 224 \
--width 224 \
--test-batch-size 100 \
--evaluate \
--print-freq 100 \
--load-weights logs/resnet50-veri/2023-03-08_best_model.pth.tar \
--save-dir logs/resnet50-eval-veri
