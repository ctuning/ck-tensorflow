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

echo
echo "Download weights from ${PACKAGE_URL} ..."
wget ${PACKAGE_URL}/${PACKAGE_NAME}

echo
echo "Unpack weights file ${PACKAGE_NAME} ..."
tar -zxvf ${PACKAGE_NAME}

echo
echo "Remove temporary files ..."
rm ${PACKAGE_NAME}

echo
echo "Copy Python modules ..."
THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cp ${THIS_SCRIPT_DIR}/tf-mobilenet-model.py .
cp ${THIS_SCRIPT_DIR}/mobilenet_v1.py .
