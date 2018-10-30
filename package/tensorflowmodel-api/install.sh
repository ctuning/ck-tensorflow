#! /bin/bash

# CK installation script for TensorFlow models
#
# Developer(s):
#  * Vladislav Zaborovskiy, vladzab@yandex.ru
#

# PACKAGE_DIR
# INSTALL_DIR
# TENSORFLOW_MODELS_URL

echo ""
echo "Removing everything in '${INSTALL_DIR}' ..."
rm -rf $INSTALL_DIR/*

######################################################################################
echo ""
echo "Downloading TensorFlow Models API into '${INSTALL_DIR}' ..."
git clone ${TENSORFLOW_MODELS_URL} ${INSTALL_DIR}

if [ "${?}" != "0" ] ; then
  echo "Error: Downloading TensorFlow Models API failed!"
  exit 1
fi

######################################################################################
echo ""
echo "Compiling Protobuf... "
cd ${INSTALL_DIR}/research/
protoc object_detection/protos/*.proto --python_out=./

if [ "${?}" != "0" ] ; then
  echo "Error: Compiling Protobuf failed!"
  exit 1
fi
