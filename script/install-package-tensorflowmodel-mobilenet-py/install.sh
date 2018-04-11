#! /bin/bash

#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
# MobileNet for TensorFlow
# Python model and weights install script
#

MULTIPLIER=${MODEL_MOBILENET_MULTIPLIER}
RESOLUTION=${MODEL_MOBILENET_RESOLUTION}
VERSION=${MODEL_MOBILENET_VERSION}

########################################################################
echo
echo "Download weights from ${PACKAGE_URL} ..."
wget ${PACKAGE_URL}/${PACKAGE_NAME}

########################################################################
echo
echo "Unpack weights file ${PACKAGE_NAME} ..."
tar -zxvf ${PACKAGE_NAME}

########################################################################
echo
echo "Remove temporary files ..."
rm ${PACKAGE_NAME}

# We don't use it right now, so remove to save disk space, but it can be useful in future
if [ ${VERSION} == "2" ]; then
  rm "mobilenet_v2_${MULTIPLIER}_${RESOLUTION}_eval.pbtxt"
  rm "mobilenet_v2_${MULTIPLIER}_${RESOLUTION}_frozen.pb"
  rm "mobilenet_v2_${MULTIPLIER}_${RESOLUTION}.tflite"
fi

########################################################################
echo
echo "Copy Python modules ..."
THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cp ${THIS_SCRIPT_DIR}/tf-mobilenet-model.py .

if [ ${VERSION} == "1" ]; then
  cp ${THIS_SCRIPT_DIR}/mobilenet_v1.py .
fi
if [ ${VERSION} == "2" ]; then
  cp ${THIS_SCRIPT_DIR}/mobilenet_v2.py .
  cp ${THIS_SCRIPT_DIR}/mobilenet.py .
  cp ${THIS_SCRIPT_DIR}/conv_blocks.py .
fi
