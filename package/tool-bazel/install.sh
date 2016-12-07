#! /bin/bash

#
# Installation script for Caffe.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developer(s):
# - Anton Lokhmotov, anton@dividiti.com, 2016
# - Grigori Fursin, grigori@dividiti.com, 2016

# PACKAGE_DIR
# INSTALL_DIR

################################################################################
echo ""
echo "Cloning from '${PACKAGE_URL}' ..."

cd ${INSTALL_DIR}
git clone ${PACKAGE_URL} src

################################################################################
echo ""
echo "Compiling bazel ..."

cd src
./compile.sh
if [ "${?}" != "0" ] ; then
  echo "Error: compiling bazel failed!"
  exit 1
fi

################################################################################
echo ""
echo "Building bazel ..."

cd output
./bazel build //scripts:bazel-complete.bash
if [ "${?}" != "0" ] ; then
  echo "Error: building bazel failed!"
  exit 1
fi

exit 0
