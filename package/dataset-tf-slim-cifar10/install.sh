#! /bin/bash

#
# Installation script for the 2012 ImageNet Large Scale Visual Recognition
# Preparing (ILSVRC'12) train dataset for TensorFlow
#
# See CK LICENSE for licensing details.
# See CK COPYRIGHT for copyright details.
#
# Developer(s):
# - Grigori Fursin, Grigori.Fursin@cTuning.org, 2018

# PACKAGE_DIR
# INSTALL_DIR

mkdir ${INSTALL_DIR}/install

cd ${CK_ENV_TENSORFLOW_MODELS}/../research/slim

${CK_ENV_COMPILER_PYTHON_FILE} download_and_convert_data.py --dataset_name=cifar10 --dataset_dir="${INSTALL_DIR}/install"

if [ "${?}" != "0" ] ; then
  echo "Error: dataset installation failed!"
  exit 1
fi

exit 0
