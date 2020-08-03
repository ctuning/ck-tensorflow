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


keep_model_filename=${PACKAGE_NAME_EFFICIENTNET_LITE_WITH_PRECISION}.tflite
echo "Keeping file ${keep_model_filename}"
for model_filename in ${PACKAGE_NAME_EFFICIENTNET_LITE}-*.tflite
do
  if [ "$model_filename" != "$keep_model_filename" ]; then
    echo "Removing file ${model_filename}"
    rm "$model_filename"
  fi
done

########################################################################
if [ "$PACKAGE_KEEP_ARCHIVE" != "YES" ]; then
    echo
    echo "Remove temporary files ..."
    rm ${PACKAGE_NAME}
fi
