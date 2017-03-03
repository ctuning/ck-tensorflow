#! /bin/bash

# CK installation script for TensorFlow model squeezeDet package
# For model Licence look LICENCE.md file in squeezeDet package folder
#
# Developer(s): 
#  * Vladislav Zaborovskiy, vladzab@yandex.ru
#

# PACKAGE_DIR
# INSTALL_DIR
# SQUEEZEDET_URL
#
# DEMO_URL
#
# SQUEEZENET_INCLUDED
# CNN_SQUEEZENET_URL
# RESNET50_INCLUDED
# CNN_RESNET50_URL
# VGG16_INCLUDED
# CNN_VGG16_URL
#
# TENSORFLOW_URL

export SQUEEZEDET_INSTALL_DIR=${INSTALL_DIR}/squeezeDet
export SQDT_ROOT=${SQUEEZEDET_INSTALL_DIR}
export SQDT_DATA=$SQDT_ROOT/data

######################################################################################
echo ""
echo "Removing everything from '${INSTALL_DIR}' ..."
rm -rf ${INSTALL_DIR}/*
mkdir $SQUEEZEDET_INSTALL_DIR

######################################################################################
echo ""
echo "Cloning SqueezeDet from '${SQUEEZEDET_URL}' to '${SQUEEZEDET_INSTALL_DIR}' ..."
git clone ${SQUEEZEDET_URL}  ${SQUEEZEDET_INSTALL_DIR}
if [ "${?}" != "0" ] ; then
  echo "Error: Cloning SqueezeDet from '${SQUEEZEDET_URL}' failed!"
  exit 1
fi

######################################################################################
echo ""
echo "Downloading demo model parameters to '${SQDT_DATA}' ..."
wget -P $SQDT_DATA ${DEMO_URL} 
if [ "${?}" != "0" ] ; then
  echo "Error: Downloading demo archive from '${DEMO_URL}' failed!"
  exit 1
fi

######################################################################################
echo ""
echo "Extracting demo archive '${SQDT_DATA}'/model_checkpoints.tgz ..."
tar -xzvf ${SQDT_DATA}/model_checkpoints.tgz -C ${SQDT_DATA}/
if [ "${?}" != "0" ] ; then
  echo "Error: Extracting demo archive '${SQDT_DATA}'/model_checkpoints.tgz failed!"
  exit 1
fi
rm ${SQDT_DATA}/model_checkpoints.tgz

######################################################################################
echo ""
echo "Downloading pretrained CNN models to '${SQDT_DATA}'/ ..."


if [ "$SQUEEZENET_INCLUDED" != "0" ] ; then
    echo "SqueezeNet"
    wget -P $SQDT_DATA $CNN_SQUEEZENET_URL
    if [ "${?}" != "0" ] ; then
        echo "Error: Downloading pretrained SqueezeNet CNN  archive from '${CNN_SQUEEZENET_URL}' failed! You may change SQUEEZENET_INCLUDED parameter in tensorflowmodel-squeezedet/.cm/meta.json file to 0 (not to dowload CNN) "
        exit 1
    fi

    echo ""
    echo "Extracting '$SQDT_DATA'/SqueezeNet.tgz ..."
    tar -xzvf ${SQDT_DATA}/SqueezeNet.tgz -C ${SQDT_DATA}/
    if [ "${?}" != "0" ] ; then
        echo "Error: Extracting CNN archive '${SQDT_DATA}'/SqueezeNet.tgz failed!"
        exit 1
    fi
    
    rm ${SQDT_DATA}/SqueezeNet.tgz
fi


if [ "$RESNET50_INCLUDED" != "0" ] ; then
    echo ""
    echo "ResNet50"
    wget -P $SQDT_DATA $CNN_RESNET50_URL
    if [ "${?}" != "0" ] ; then
        echo "Error: Downloading pretrained ResNet50 CNN  archive from '${CNN_SQUEEZENET_URL}' failed! You may change SQUEEZENET_INCLUDED parameter in tensorflowmodel-squeezedet/.cm/meta.json file to 0 (not to dowload CNN)"
        exit 1
    fi
    
    echo ""
    echo "Extracting '$SQDT_DATA'/ResNet.tgz ..."
    tar -xzvf ${SQDT_DATA}/ResNet.tgz -C ${SQDT_DATA}/
    if [ "${?}" != "0" ] ; then
        echo "Error: Extracting CNN archive '${SQDT_DATA}'/ResNet.tgz failed!"
        exit 1
    fi
    
    rm ${SQDT_DATA}/ResNet.tgz
fi

if [ "$VGG16_INCLUDED" != "0" ] ; then
    echo ""
    echo "VGG16"
    wget -P $SQDT_DATA $CNN_VGG16_URL
    if [ "${?}" != "0" ] ; then
        echo "Error: Downloading pretrained VGG16 CNN  archive from '${CNN_SQUEEZENET_URL}' failed! You may change SQUEEZENET_INCLUDED parameter in tensorflowmodel-squeezedet/.cm/meta.json file to 0 (not to dowload CNN)"
        exit 1
    fi
    
    echo ""
    echo "Extracting '$SQDT_DATA'/VGG16.tgz ..."
    tar -xzvf ${SQDT_DATA}/VGG16.tgz -C ${SQDT_DATA}/
    if [ "${?}" != "0" ] ; then
        echo "Error: Extracting CNN archive '${SQDT_DATA}'/VGG16.tgz failed!"
        exit 1
    fi
    
    rm ${SQDT_DATA}/VGG16.tgz
fi

######################################################################################
echo ""
echo "Downloading TensorFlow upgrade scripts(to version 1.x from 0.x)..."

#TODO Download subdir from tensorflow repository without 3d party software
git clone $TENSORFLOW_URL ${INSTALL_DIR}/tensorflow_update
if [ "${?}" != "0" ] ; then
  echo "Error: Downloading TensorFlow from '${TENSORFLOW_URL}' failed!"
  exit 1
fi
export TENSORFLOW_UPDATE_PATH=${INSTALL_DIR}/tensorflow_update/tensorflow/tools/compatibility

echo ""
echo "Upgrading TensorFlow scripts(to version 1.x from 0.x)..."

$CK_ENV_COMPILER_PYTHON_FILE $TENSORFLOW_UPDATE_PATH/tf_upgrade.py --intree $SQDT_ROOT --outtree ${INSTALL_DIR}/squeezeDet_updated

cp -rf ${INSTALL_DIR}/squeezeDet_updated/* ${SQDT_ROOT}

######################################################################################
#Some manual text editing
echo ""
echo "Small source code change ..."
sed -i 's/max_images/max_outputs/g' ${SQDT_ROOT}/src/nn_skeleton.py

######################################################################################
echo ""
echo "Cleaning ..."
rm -rf ${INSTALL_DIR}/squeezeDet_updated
rm -rf ${INSTALL_DIR}/tensorflow_update

exit 0

