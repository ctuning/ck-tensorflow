#! /bin/bash

#
# Copyright (c) 2018-2020 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
# EfficientNet-Lite for TensorFlow
# Python model and weights install script

########################################################################
echo
echo "Download weights from ${PACKAGE_URL} ..."
wget ${PACKAGE_URL}/${PACKAGE_NAME}

########################################################################
echo
echo "Unpack weights file ${PACKAGE_NAME} ..."
tar -zxvf ${PACKAGE_NAME}

# EfficientNet-Lite unpacks into subdirectory
echo
echo "Move files out of ${PACKAGE_NAME_EFFICIENTNET_LITE}/ ..."
mv ${PACKAGE_NAME_EFFICIENTNET_LITE}/* ${PACKAGE_NAME_EFFICIENTNET_LITE}/..
rmdir ${PACKAGE_NAME_EFFICIENTNET_LITE}
echo "Keeping file ${PACKAGE_NAME_EFFICIENTNET_LITE_WITH_PRECISION}.tflite"
if [ "${MODEL_EFFICIENTNET_LITE_PRECISION}" == "fp32" ]; then
  file_to_remove="${PACKAGE_NAME_EFFICIENTNET_LITE}-int8.tflite"
  echo "Removing file ${file_to_remove}"
  rm ${file_to_remove}
else
  file_to_remove="${PACKAGE_NAME_EFFICIENTNET_LITE}-fp32.tflite"
  echo "Removing file ${file_to_remove}"
  rm ${file_to_remove}
fi

########################################################################
echo
echo "Remove temporary files ..."
rm ${PACKAGE_NAME}
