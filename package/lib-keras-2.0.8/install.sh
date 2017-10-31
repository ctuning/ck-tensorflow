#! /bin/bash

# CK installation script for Keras package
#
# Developer(s):
#  * Vladislav Zaborovskiy, vladzab@yandex.ru
#
# INSTALL_DIR

rm -rf $INSTALL_DIR/*
######################################################################################
echo ""
echo "Downloading and installing YAML..."
$CK_PYTHON_PIP_BIN install pyyaml -t $INSTALL_DIR
if [ "${?}" != "0" ] ; then
    echo ""
    echo "Error: pyyaml installation failed!"
    exit 1
fi

######################################################################################
echo ""
echo "Downloading and installing Keras..."
$CK_PYTHON_PIP_BIN install keras -t $INSTALL_DIR
if [ "${?}" != "0" ] ; then
    echo ""
    echo "Error: Keras pip installation failed failed!"
    exit 1
fi

######################################################################################
echo ""
echo "Successfully finished Keras installation."
echo "Warning: ck-keras strongly depends on chosen TensorFlow version. In case you delete TF library, delete this soft as well and reinstall"
exit 0
