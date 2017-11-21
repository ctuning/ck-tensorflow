#! /bin/bash

# CK installation script for TensorFlow package
#
# Developer(s):
#  * Grigori Fursin, dividiti/cTuning foundation
#

# PACKAGE_DIR
# INSTALL_DIR
# TENSORFLOW_URL

export TENSORFLOW_LIB_DIR=${INSTALL_DIR}/lib

######################################################################################
echo ""
echo "Removing '${TENSORFLOW_LIB_DIR}' ..."
rm -rf ${TENSORFLOW_LIB_DIR}

######################################################################################
# Print info about possible issues
echo ""
echo "Note that you sometimes need to upgrade your pip to the latest version"
echo "to avoid well-known issues with user/system space installation:"
echo ""
echo "sudo ${CK_PYTHON_PIP_BIN} install --upgrade pip"
echo ""
read -p "Press enter to continue"

SUDO="sudo "
if [[ ${CK_PYTHON_PIP_BIN_FULL} == /home/* ]] ; then
  SUDO=""
fi

# Check if has --system option
${CK_PYTHON_PIP_BIN_FULL} install --help > tmp-pip-help.tmp
if grep -q "\-\-system" tmp-pip-help.tmp ; then
 SYS=" --system"
fi
rm -f tmp-pip-help.tmp

######################################################################################
echo ""
echo "Downloading and installing ProtoBuf (${PROTOBUF_PYTHON_URL}) ..."
echo ""

${CK_PYTHON_PIP_BIN_FULL} install --upgrade ${PROTOBUF_PYTHON_URL} -t ${TENSORFLOW_LIB_DIR} --trusted-host storage.googleapis.com --trusted-host pypi.python.org ${SYS}
######################################################################################
if [ "${?}" != "0" ] ; then
  echo "Error: installation failed!"
  exit 1
fi

echo ""
echo "Downloading and installing TensorFlow prebuilt binaries (${TF_PYTHON_URL}) ..."
echo ""

${CK_PYTHON_PIP_BIN_FULL} install --upgrade ${TF_PYTHON_URL} -t ${TENSORFLOW_LIB_DIR} --trusted-host storage.googleapis.com ${SYS}
######################################################################################
if [ "${?}" != "0" ] ; then
  echo "Error: installation failed!"
  exit 1
fi

exit 0
