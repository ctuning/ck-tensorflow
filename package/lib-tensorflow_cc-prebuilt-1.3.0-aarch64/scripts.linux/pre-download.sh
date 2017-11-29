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

ELF32=`file -L /sbin/init | grep -o "ELF 32-bit"`
ELF64=`file -L /sbin/init | grep -o "ELF 64-bit"`

if [ -n "$ELF32" ]; then
    export PACKAGE_NAME="$PACKAGE_NAME_32_BIT"
elif [ -n "$ELF64" ]; then
    export PACKAGE_NAME="$PACKAGE_NAME_64_BIT"
else
    echo "Unsupported executable format. Only ELF 32-bit and ELF 64-bit are supported for now"
    exit 1
fi

export PACKAGE_NAME1="$PACKAGE_NAME"
