#! /bin/bash

# CK installation script for TensorFlow package
#
# Developer(s):
#  * Vladislav Zaborovskiy, vladzab@yandex.ru
#

# PACKAGE_DIR
# INSTALL_DIR
# TENSORFLOW_MODELS_URL

echo ""
echo "Removing everything in '${INSTALL_DIR}' ..."
cd $INSTALL_DIR
rm -rf *

######################################################################################
echo ""
echo "Downloading Tensorflow Models api into '${INSTALL_DIR}' ..."
git clone $TENSORFLOW_MODELS_URL $INSTALL_DIR

if [ "${?}" != "0" ] ; then
  echo "Error: Downloading Tensorflow Models api failed!"
  exit 1
fi

######################################################################################
echo ""
echo "Protobuf compilation... "
cd $INSTALL_DIR
protoc object_detection/protos/*.proto --python_out=./

if [ "${?}" != "0" ] ; then
  echo "Error: Protobuf compilation failed!"
  exit 1
fi
