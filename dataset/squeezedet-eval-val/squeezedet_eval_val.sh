#!/bin/bash

echo ""
echo "0) squeezeDet"
echo "1) squeezeDetPlus"
echo "2) vgg16"
echo "3) ResNet50"

read -p "Choose CNN pretrained model[default 0]: " choice
choice=${choice:-0}

cd $CK_ENV_MODEL_SQUEEZEDET_ROOT
# =========================================================================== #
# command for squeezeDet:
# =========================================================================== #
if [ "$choice" = "0" ] ; then 
python ./src/eval.py \
  --dataset=KITTI \
  --data_path=./data/KITTI \
  --image_set=val \
  --eval_dir=/tmp/bichen/logs/SqueezeDet/eval_val \
  --checkpoint_path=/tmp/bichen/logs/SqueezeDet/train \
  --net=squeezeDet \
  --gpu=0
fi
# =========================================================================== #
# command for squeezeDet+:
# =========================================================================== 
if [ "$choice" = "1" ] ; then 
 python ./src/eval.py \
   --dataset=KITTI \
   --data_path=./data/KITTI \
   --image_set=val \
   --eval_dir=/tmp/bichen/logs/SqueezeDetPlus/eval_val \
   --checkpoint_path=/tmp/bichen/logs/SqueezeDetPlus/train \
   --net=squeezeDet+ \
   --gpu=0
fi
# =========================================================================== #
# command for vgg16:
# =========================================================================== 
if [ "$choice" = "2" ] ; then 
 python ./src/eval.py \
   --dataset=KITTI \
   --data_path=./data/KITTI \
   --image_set=val \
   --eval_dir=/tmp/bichen/logs/vgg16/eval_val \
   --checkpoint_path=/tmp/bichen/logs/vgg16/train \
   --net=squeezeDet+ \
   --gpu=0
fi
# =========================================================================== #
# command for resnet50:
# =========================================================================== 
if [ "$choice" = "3" ] ; then 
 python ./src/eval.py \
   --dataset=KITTI \
   --data_path=./data/KITTI \
   --image_set=val \
   --eval_dir=/tmp/bichen/logs/resnet50/eval_train \
   --checkpoint_path=/tmp/bichen/logs/resnet50/train \
   --net=resnet50 \
   --gpu=0
fi
