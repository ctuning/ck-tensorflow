#! /bin/bash

# CK installation script for TensorFlow package
#
# Developer(s): 
#  * Vladislav Zaborovskiy, vladzab@yandex.ru
#  * Grigori Fursin, dividiti/cTuning foundation
#

# PACKAGE_DIR
# INSTALL_DIR
# TENSORFLOW_URL

export TENSORFLOW_LIB_DIR=${INSTALL_DIR}/lib

######################################################################################
echo ""
echo "Removing everything from '${TENSORFLOW_LIB_DIR}' ..."
rm -rf ${TENSORFLOW_LIB_DIR}

######################################################################################
echo "" 
echo "Downloading and installing TensorFlow prebuilt package ..."

cd ${INSTALL_DIR}

if [[ $PYTHON3 == 0 ]]
then
    if [[ $GPU_ENABLED == 1 ]]
    then
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_PACKAGE_VER}-cp27-none-linux_x86_64.whl 
    else
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_PACKAGE_VER}-cp27-none-linux_x86_64.whl 
    fi
    pip install --upgrade $TF_BINARY_URL -t ${TENSORFLOW_LIB_DIR}
else
    if [[ $GPU_ENABLED == 1 ]]
    then
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_PACKAGE_VER}-cp35-cp35m-linux_x86_64.whl
    else
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_PACKAGE_VER}-cp35-cp35m-linux_x86_64.whl
    fi
    pip3 install --upgrade $TF_BINARY_URL -t ${TENSORFLOW_LIB_DIR}
fi

if [ "${?}" != "0" ] ; then
  echo "Error: package installation failed"
  exit 1
fi

exit 0
