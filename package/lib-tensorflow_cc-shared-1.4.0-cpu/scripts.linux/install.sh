#! /bin/bash

#
# Extra installation script.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developer(s):
# - Grigori Fursin, grigori.fursin@cTuning.org, 2017
# - Flavio Vella, flavio@dividiti.com, 2017
# - Anton Lokhmotov, anton@dividiti.com, 2017
#

cp ${ORIGINAL_PACKAGE_DIR}/*.patch ${INSTALL_DIR}/obj

export JOB_COUNT=${CK_HOST_CPU_NUMBER_OF_PROCESSORS}

export CK_CMAKE_EXTRA="${CK_CMAKE_EXTRA} \
  -DTENSORFLOW_SHARED=ON \
  -DTENSORFLOW_STATIC=OFF \
  -DCMAKE_CXX_COMPILER=${CK_CXX_PATH_FOR_CMAKE} \
  -DCMAKE_CXX_FLAGS=${CK_CXX_FLAGS_FOR_CMAKE} \
  -DCMAKE_CC_COMPILER=${CK_CC_PATH_FOR_CMAKE} \
  -DCMAKE_CC_FLAGS=${CK_CC_FLAGS_FOR_CMAKE}"

return 0
