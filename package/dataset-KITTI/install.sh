#! /bin/bash

# CK installation script for KITTI dataset package, squeezeNet built-in
#
# Developer(s): 
#  * Vladislav Zaborovskiy, vladzab@yandex.ru
#  

# PACKAGE_DIR
# INSTALL_DIR

# KITTI_IMAGES_URL
# KITTI_LABELS_URL

# From deps
# CK_ENV_MODEL_SQUEEZEDET_ROOT
# CK_ENV_COMPILER_PYTHON_FILE

export SQDT_ROOT=$CK_ENV_MODEL_SQUEEZEDET_ROOT
export SQDT_DATA=${SQDT_ROOT}/data
export KITTI_ROOT=${SQDT_DATA}/KITTI
export KITTI_VAL=${KITTI_ROOT}/ImageSets

######################################################################################
echo ""
echo "Creating KITTI folders ..."
mkdir $KITTI_ROOT
mkdir $KITTI_VAL

######################################################################################
echo ""
echo "Warning:  More than 26GB of free space required."

echo "Downloading KITTI images ..."
wget -P $KITTI_ROOT $KITTI_IMAGES_URL
if [ "${?}" != "0" ] ; then
  echo "Error: Downloading KITTI images  archive from '${KITTI_IMAGES_URL}' failed!"
  exit 1
fi

echo ""
echo "Downloading KITTI labels ..."
wget -P $KITTI_ROOT $KITTI_LABELS_URL

if [ "${?}" != "0" ] ; then
  echo "Error: Downloading KITTI labels  archive from '${KITTI_LABELS_URL}' failed!"
  exit 1
fi

######################################################################################
echo ""
echo "Extracting archives ..."
echo "Extracting '$KITTI_ROOT'/data_object_image_2.zip ..."
unzip  ${KITTI_ROOT}/data_object_image_2.zip -d ${KITTI_ROOT}
if [ "${?}" != "0" ] ; then
  echo "Error: Extracting images archive '${KITTI_ROOT}'/data_object_image_2.zip failed!"
  exit 1
fi
rm ${KITTI_ROOT}/data_object_image_2.zip

echo ""
echo "Extracting '$KITTI_ROOT'/data_object_label_2.zip ..."
unzip  ${KITTI_ROOT}/data_object_label_2.zip -d ${KITTI_ROOT}
if [ "${?}" != "0" ] ; then
  echo "Error: Extracting labels archive '${KITTI_ROOT}'/data_object_label_2.zip failed!"
  exit 1
fi
rm ${KITTI_ROOT}/data_object_label_2.zip

######################################################################################
echo ""
echo "Splitting the training data into a training set and a validation set ..."
cd $KITTI_VAL
ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
cd $SQDT_DATA
$CK_ENV_COMPILER_PYTHON_FILE random_split_train_val.py
if [ "${?}" != "0" ] ; then
  echo "Error: Validation with '$SQDT_DATA'random_split_train_val.py' failed! Do it manually with $ python random_split_train_val.py"
  exit 1
fi

exit 0
