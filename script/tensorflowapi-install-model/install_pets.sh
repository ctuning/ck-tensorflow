#! /bin/bash

echo ""
echo "Generating TFRecord files for Oxford Pets dataset..."

$CK_PYTHON_BIN $CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/create_pet_tf_record.py --label_map_path=$CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/data/pet_label_map.pbtxt --data_dir=$CK_ENV_DATASET_PET --output_dir=$DATA_DIR
if [ "${?}" != "0" ] ; then
  echo "Error: Generating TFRecord files for training and validation failed!"
  exit 1
fi

cp -f $CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR/data/pet_label_map.pbtxt $DATA_DIR
if [ "${?}" != "0" ] ; then
  echo "Error: Copying label map failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Successfully created TF Record for Oxford Pets dataset ..."
exit 0
