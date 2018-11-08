#! /bin/bash

# CK installation script for TensorFlow models
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
#

echo ""
echo "Compiling Protobuf... "
cd ${INSTALL_DIR}/${PACKAGE_SUB_DIR}/research
${CK_ENV_LIB_PROTOBUF_HOST_BIN}/protoc object_detection/protos/*.proto --python_out=./

if [ "${?}" != "0" ] ; then
  echo "Error: Compiling Protobuf failed!"
  exit 1
fi
