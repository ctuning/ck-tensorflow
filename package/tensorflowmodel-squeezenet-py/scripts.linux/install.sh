#! /bin/bash

#
# Copyright (c) 2017-2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
# SqueezeNet for TensorFlow
# Python model and weights install script
#

echo 'Copy python model ...'
cp ${ORIGINAL_PACKAGE_DIR}/squeezenet-model.py .

echo 'Copy model weights ...'
cp ${ORIGINAL_PACKAGE_DIR}/tf-squeezenet-weights.mat .
