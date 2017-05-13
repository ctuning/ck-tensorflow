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
if [ "$HOSTTYPE" != "x86_64" ] ; then
  echo "Error: this package only supports x86_64!"
  exit 1
fi

######################################################################################
echo ""
echo "Removing '${TENSORFLOW_LIB_DIR}' ..."
rm -rf ${TENSORFLOW_LIB_DIR}

######################################################################################
echo ""
echo "Downloading and installing TensorFlow prebuilt binaries ..."
# cd ${INSTALL_DIR}

if [[ $PYTHON3 == 0 ]]
then
  if [[ $GPU_ENABLED == 1 ]]
  then
    export TENSORFLOW_BIN_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-${TENSORFLOW_PACKAGE_VER}-cp27-none-linux_x86_64.whl
  else
    export TENSORFLOW_BIN_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_PACKAGE_VER}-cp27-none-linux_x86_64.whl
  fi
  pip install --upgrade ${TENSORFLOW_BIN_URL} -t ${TENSORFLOW_LIB_DIR} --trusted-host storage.googleapis.com
else
  if [[ $GPU_ENABLED == 1 ]]
  then
    export TENSORFLOW_BIN_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-${TENSORFLOW_PACKAGE_VER}-cp35-cp35m-linux_x86_64.whl
  else
    export TENSORFLOW_BIN_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_PACKAGE_VER}-cp35-cp35m-linux_x86_64.whl
  fi
  pip3 install --upgrade ${TENSORFLOW_BIN_URL} -t ${TENSORFLOW_LIB_DIR} --trusted-host storage.googleapis.com
fi

######################################################################################
if [ "${?}" != "0" ] ; then
  echo "Error: installation failed!"
  exit 1
fi

exit 0
