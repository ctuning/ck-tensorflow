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
echo "Downloading models into '${INSTALL_DIR}' ..."
git clone $TENSORFLOW_MODELS_URL $INSTALL_DIR

######################################################################################
if [ "${?}" != "0" ] ; then
  echo "Error: installation failed!"
  exit 1
fi

