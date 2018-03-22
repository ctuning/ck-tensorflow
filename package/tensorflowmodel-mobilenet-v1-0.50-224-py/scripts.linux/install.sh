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

echo 'Copy python model ...'
cp ${ORIGINAL_PACKAGE_DIR}/tf-mobilenet-model.py .
cp ${ORIGINAL_PACKAGE_DIR}/mobilenet_v1.py .
