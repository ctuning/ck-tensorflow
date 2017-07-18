#! /bin/bash

# PACKAGE_DIR
# INSTALL_DIR

rm -rf $INSTALL_DIR/*
mkdir $INSTALL_DIR/data
export DATA_DIR=$INSTALL_DIR/data
cd $DATA_DIR
export MODEL_DIR=$INSTALL_DIR/model
export TRAIN_DIR=$MODEL_DIR/train
export EVAL_DIR=$MODEL_DIR/eval
mkdir $MODEL_DIR
mkdir $EVAL_DIR
mkdir $TRAIN_DIR

#####################################################################
echo ""
echo "Generating TFRecord files for training and validation... "

#from $INSTALL_DIR/data
$CK_PYTHON_BIN $CK_ENV_TENSORFLOW_MODELS_ROOT/create_pascal_tf_record.py --label_map_path=$CK_ENV_TENSORFLOW_MODELS_ROOT/data/pascal_label_map.pbtxt --data_dir=$CK_ENV_DATASET_VOC/VOCdevkit --year=VOC2012 --set=train --output_path=pascal_train.record

$CK_PYTHON_BIN $CK_ENV_TENSORFLOW_MODELS_ROOT/create_pascal_tf_record.py --label_map_path=$CK_ENV_TENSORFLOW_MODELS_ROOT/data/pascal_label_map.pbtxt --data_dir=$CK_ENV_DATASET_VOC/VOCdevkit --year=VOC2012 --set=val --output_path=pascal_val.record

cp -f $CK_ENV_TENSORFLOW_MODELS_ROOT/data/pascal_label_map.pbtxt ./

if [ "${?}" != "0" ] ; then
  echo "Error: Generating TFRecord files for training and validation failed!"
  exit 1
fi

######################################################################################
echo ""
echo "Generating pipeline.config file ..."
cd $PACKAGE_DIR
cp -f pipeline.template pipeline.config
sed -i "s|_PATH_TO_CHECKPOINT_|$TRAIN_DIR|g" pipeline.config
sed -i "s|_PATH_TO_INPUT_|$DATA_DIR|g" pipeline.config
sed -i "s|_PATH_TO_LABEL_MAP_|$DATA_DIR|g" pipeline.config
cp -f pipeline.config $MODEL_DIR/

if [ "${?}" != "0" ] ; then
  echo "Error: Generating pipeline.config file failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Successfully installed Faster RCNN resnet101 tensorflow model ..."
exit 0


exit 0

