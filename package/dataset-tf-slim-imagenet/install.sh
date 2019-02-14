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

bazel build :download_and_convert_imagenet
if [ "${?}" != "0" ] ; then
  echo "Error: dataset script preparation failed!"
  exit 1
fi

./bazel-bin/download_and_convert_imagenet "${INSTALL_DIR}/install"
if [ "${?}" != "0" ] ; then
  echo "Error: dataset installation failed!"
  exit 1
fi

exit 0
