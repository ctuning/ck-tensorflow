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

# Exception: v2 quantized unpacks into a subdirectory.
if [[ -d ${PACKAGE_NAME_V2_QUANT} ]]; then
  echo
  echo "Move files out of ${PACKAGE_NAME_V2_QUANT}/ ..."
  mv ${PACKAGE_NAME_V2_QUANT}/* ${PACKAGE_NAME_V2_QUANT}/..
  rmdir ${PACKAGE_NAME_V2_QUANT}
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
rm_file "mobilenet_v${VERSION}_${MULTIPLIER}_${RESOLUTION}_eval.pbtxt"

########################################################################
echo
echo "Copy Python modules ..."
THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cp ${THIS_SCRIPT_DIR}/mobilenet-model.py .

if [ "${VERSION}" == "1" ]; then
  cp ${THIS_SCRIPT_DIR}/mobilenet_v1.py .
fi
if [ "${VERSION}" == "2" ]; then
  cp ${THIS_SCRIPT_DIR}/mobilenet_v2.py .
  cp ${THIS_SCRIPT_DIR}/mobilenet.py .
  cp ${THIS_SCRIPT_DIR}/conv_blocks.py .
fi
