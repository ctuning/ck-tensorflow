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
export PIPELINE_NAME="faster_rcnn_resnet101_voc"
export MODEL_NAME="Faster RCNN resnet101 with VOC dataset"

#####################################################################
echo ""
echo "Generating TFRecord files for training and validation... "

#from $INSTALL_DIR/data
$CK_PYTHON_BIN $CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/create_pascal_tf_record.py --label_map_path=$CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/data/pascal_label_map.pbtxt --data_dir=$CK_ENV_DATASET_VOC/train/VOCdevkit --year=VOC2007 --set=train --output_path=pascal_train.record

$CK_PYTHON_BIN $CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/create_pascal_tf_record.py --label_map_path=$CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/data/pascal_label_map.pbtxt --data_dir=$CK_ENV_DATASET_VOC/train/VOCdevkit --year=VOC2007 --set=val --output_path=pascal_val.record

if [ "${?}" != "0" ] ; then
  echo "Error: Generating TFRecord files for training and validation failed!"
  exit 1
fi

cp -f $CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/data/pascal_label_map.pbtxt ./

######################################################################################
echo ""
echo "Generating '${PIPELINE_NAME}'.config file ..."
cd $PACKAGE_DIR
cp -f $PIPELINE_NAME.template $PIPELINE_NAME.config
sed -i "s|_PATH_TO_CHECKPOINT_|$TRAIN_DIR|g" $PIPELINE_NAME.config
sed -i "s|_PATH_TO_INPUT_|$DATA_DIR|g" $PIPELINE_NAME.config
sed -i "s|_PATH_TO_LABEL_MAP_|$DATA_DIR|g" $PIPELINE_NAME.config
cp -f $PIPELINE_NAME.config $MODEL_DIR/
rm $PIPELINE_NAME.config

if [ "${?}" != "0" ] ; then
  echo "Error: Generating pipeline.config file failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Successfully installed '${MODEL_NAME}' tensorflow model ..."
exit 0


exit 0

