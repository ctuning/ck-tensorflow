#! /bin/bash

#
# Copyright (c) 2018-2020 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
# MobileNet for TensorFlow
# Python model and weights install script
#

########################################################################
echo
echo "Download weights from ${PACKAGE_URL} ..."
wget ${PACKAGE_URL}/${PACKAGE_NAME}

########################################################################
echo
echo "Unpack weights file ${PACKAGE_NAME} ..."
tar -zxvf ${PACKAGE_NAME}


#Efficientnet-lite unpacks into subdirectory
if [[ -d ${PACKAGE_NAME_EFFICIENTNET_LITE} ]]; then
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


fi

########################################################################
echo
echo "Remove temporary files ..."
rm ${PACKAGE_NAME}

function rm_file() {
  if [ -f $1 ]; then
    rm $1
  fi
}
# We don't use it right now, so remove to save disk space, but it can be useful in future
#rm_file "mobilenet_v${VERSION}_${MULTIPLIER}_${RESOLUTION}_eval.pbtxt"

########################################################################
echo
echo "Copy Python modules ..."
THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#cp ${THIS_SCRIPT_DIR}/mobilenet-model.py .

#if [ "${VERSION}" == "1" ]; then
#  cp ${THIS_SCRIPT_DIR}/mobilenet_v1.py .
#fi
#if [ "${VERSION}" == "2" ]; then
#  cp ${THIS_SCRIPT_DIR}/mobilenet_v2.py .
#  cp ${THIS_SCRIPT_DIR}/mobilenet.py .
#  cp ${THIS_SCRIPT_DIR}/conv_blocks.py .
#fi
#if [ "${VERSION}" == "3" ]; then
#  cp ${THIS_SCRIPT_DIR}/mobilenet_v3.py .
#  cp ${THIS_SCRIPT_DIR}/mobilenet.py .
#  cp ${THIS_SCRIPT_DIR}/conv_blocks.py .
#fi
