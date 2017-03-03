#! /bin/bash

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
$CK_ENV_COMPILER_PYTHON_FILE ./src/train.py \
  --dataset=KITTI \
  --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl \
  --data_path=./data/KITTI \
  --image_set=train \
  --train_dir=/tmp/bichen/logs/SqueezeDet/train \
  --net=squeezeDet \
  --summary_step=100 \
  --checkpoint_step=500 \
  --gpu=0
fi
# =========================================================================== #
# command for squeezeDet+:
# =========================================================================== #
if [ "$choice" = "1" ] ; then
 $CK_ENV_COMPILER_PYTHON_FILE ./src/train.py \
   --dataset=KITTI \
   --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.0_SR_0.750.pkl \
   --data_path=./data/KITTI \
   --image_set=train \
   --train_dir=/tmp/bichen/logs/SqueezeDetPlus/train \
   --net=squeezeDet+ \
   --summary_step=100 \
   --checkpoint_step=500 \
   --gpu=0
fi
# =========================================================================== #
# command for vgg16:
# =========================================================================== #
if [ "$choice" = "2" ] ; then
 $CK_ENV_COMPILER_PYTHON_FILE ./src/train.py \
   --dataset=KITTI \
   --pretrained_model_path=./data/VGG16/VGG_ILSVRC_16_layers_weights.pkl \
   --data_path=./data/KITTI \
   --image_set=train \
   --train_dir=/tmp/bichen/logs/vgg16/train \
   --net=vgg16 \
   --summary_step=100 \
   --checkpoint_step=500 \
   --gpu=0
fi
# =========================================================================== #
# command for resnet50:
# =========================================================================== #
if [ "$choice" = "3" ] ; then
 $CK_ENV_COMPILER_PYTHON_FILE ./src/train.py \
   --dataset=KITTI \
   --pretrained_model_path=./data/ResNet/ResNet-50-weights.pkl \
   --data_path=./data/KITTI \
   --image_set=train \
   --train_dir=/tmp/bichen/logs/resnet/train \
   --net=resnet50 \
   --summary_step=100 \
   --checkpoint_step=500 \
   --gpu=0
fi
