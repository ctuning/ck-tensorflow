#! /bin/bash

#
# Copyright (c) 2015-2017 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

ARCH=`uname -m`

if [ "$ARCH" != "aarch64" ]; then 
    echo "Unsupported architecture. Only aarch64 is supported for now"
    exit 1
fi
