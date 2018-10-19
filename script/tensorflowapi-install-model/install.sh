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
if [ $DATASET_NAME == 'coco' ] || [ $DATASET_NAME == 'pets' ] || [ $DATASET_NAME == 'voc' ] || [ $DATASET_NAME == 'kitti' ];
then
    echo ""
    echo "Running install_$DATASET_NAME.sh script."
    sh $PACKAGE_DIR/install_$DATASET_NAME.sh
else
    echo ""
    echo "Error: Set Dataset name parameter from ['pets', 'voc', 'coco', 'kitti'] in your meta file."
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

########################################################################
cd $INSTALL_DIR
echo
echo "Download weights from ${PACKAGE_URL} ..."
wget ${PACKAGE_URL}/${PACKAGE_NAME}

########################################################################
echo
echo "Unpack weights file ${PACKAGE_NAME} ..."
tar -zxvf ${PACKAGE_NAME}

########################################################################
echo
echo "Remove temporary files ..."
rm $INSTALL_DIR/$PACKAGE_NAME
mv $INSTALL_DIR/$PACKAGE_NAME1/$FROZEN_GRAPH  $MODEL_DIR
if [ $WEIGHTS_FILE ]; then
  if [ $MODEL_WEIGHTS_ARE_CHECKPOINTS = "YES" ]; then
    mv $INSTALL_DIR/$PACKAGE_NAME1/$WEIGHTS_FILE*  $MODEL_DIR
  else
    mv $INSTALL_DIR/$PACKAGE_NAME1/$WEIGHTS_FILE  $MODEL_DIR
  fi
fi
rm -r $INSTALL_DIR/$PACKAGE_NAME1

############################################################
if [ -f "${ORIGINAL_PACKAGE_DIR}/scripts.${CK_TARGET_OS_ID}/install.sh" ] ; then
  echo ""
  echo "Executing extra script ..."

  . ${ORIGINAL_PACKAGE_DIR}/scripts.${CK_TARGET_OS_ID}/install.sh $MODEL_DIR

  if [ "${?}" != "0" ] ; then
    echo "Error: Failed executing extra script ..."
    exit 1
  fi
fi

#####################################################################
echo ""
echo "Successfully installed '${MODEL_NAME}' tensorflow model ..."
exit 0
