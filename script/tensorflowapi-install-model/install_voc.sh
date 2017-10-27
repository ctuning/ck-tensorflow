#! /bin/bash

echo ""
echo "Generating TFRecord files for VOC2007 dataset.. "

$CK_PYTHON_BIN $CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/create_pascal_tf_record.py --label_map_path=$CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/data/pascal_label_map.pbtxt --data_dir=$CK_ENV_DATASET_VOC/train/VOCdevkit --year=VOC2007 --set=train --output_path=$DATA_DIR/pascal_train.record
if [ "${?}" != "0" ] ; then
  echo "Error: Generating TFRecord files for VOC2007 dataset!"
  exit 1
fi

$CK_PYTHON_BIN $CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/create_pascal_tf_record.py --label_map_path=$CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/data/pascal_label_map.pbtxt --data_dir=$CK_ENV_DATASET_VOC/train/VOCdevkit --year=VOC2007 --set=val --output_path=$DATA_DIR/pascal_val.record
if [ "${?}" != "0" ] ; then
  echo "Error: Generating TFRecord files for VOC2007 dataset!"
  exit 1
fi

cp -f $CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/data/pascal_label_map.pbtxt $DATA_DIR
if [ "${?}" != "0" ] ; then
  echo "Error: Copying TF Record file failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Successfully created TF Record file for VOC2007 dataset."
exit 0
