#! /bin/bash

# PACKAGE_DIR
# INSTALL_DIR
# PIPELINE_NAME
# MODEL_NAME
# DATASET_NAME


#####################################################################
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
if [ $DATASET_NAME == 'coco' ] || [ $DATASET_NAME == 'pets' ] || [ $DATASET_NAME == 'voc' ];
then
    echo ""
    echo "Runnin install_$DATASET_NAME.sh script."
    sh $PACKAGE_DIR/install_$DATASET_NAME.sh
else
    echo ""
    echo "Error: Set Dataset name parameter from ['pets', 'voc', 'coco'] in your meta file."
    exit 1
fi

#####################################################################
echo ""
echo "Generating '${PIPELINE_NAME}'.config file ..."
cd $PACKAGE_DIR/configs
cp -f $PIPELINE_NAME.template $PIPELINE_NAME.config
sed -i "s|_PATH_TO_CHECKPOINT_|$TRAIN_DIR|g" $PIPELINE_NAME.config
sed -i "s|_PATH_TO_INPUT_|$DATA_DIR|g" $PIPELINE_NAME.config
sed -i "s|_PATH_TO_LABEL_MAP_|$DATA_DIR|g" $PIPELINE_NAME.config
cp -f $PIPELINE_NAME.config $MODEL_DIR/
rm $PIPELINE_NAME.config

if [ "${?}" != "0" ] ; then
  echo "Error: Generating '${PIPELINE_NAME}'.config file failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Successfully installed '${MODEL_NAME}' tensorflow model ..."
exit 0
