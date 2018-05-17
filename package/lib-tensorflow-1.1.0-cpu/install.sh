#! /bin/bash

# CK installation script for TensorFlow package
#
# Developer(s):
#  * Grigori Fursin, dividiti/cTuning foundation
#

# PACKAGE_DIR
# INSTALL_DIR
# PROTOBUF_PYTHON_URL
# TF_PYTHON_URL

    # This is where pip will install the modules.
    # It has its own funny structure we don't control :
    #
EXTRA_PYTHON_SITE=${INSTALL_DIR}/python_deps_site

SHORT_PYTHON_VERSION=`${CK_ENV_COMPILER_PYTHON_FILE} -c 'import sys;print(sys.version[:3])'`
export PACKAGE_LIB_DIR="${EXTRA_PYTHON_SITE}/lib/python${SHORT_PYTHON_VERSION}/site-packages"
export PYTHONPATH=$PACKAGE_LIB_DIR:$PYTHONPATH

######################################################################################
echo ""
echo "Removing '${EXTRA_PYTHON_SITE}' ..."
rm -rf ${EXTRA_PYTHON_SITE}

######################################################################################
# Check if has --system option
${CK_ENV_COMPILER_PYTHON_FILE} -m pip install --help > tmp-pip-help.tmp
if grep -q "\-\-system" tmp-pip-help.tmp ; then
 SYS=" --system"
fi
rm -f tmp-pip-help.tmp

######################################################################################
echo "Downloading and installing ProtoBuf (${PROTOBUF_PYTHON_URL}) and TensorFlow (${TF_PYTHON_URL}) prebuilt libraries..."
echo ""

${CK_ENV_COMPILER_PYTHON_FILE} -m pip install --ignore-installed ${PROTOBUF_PYTHON_URL} ${TF_PYTHON_URL} --prefix=${EXTRA_PYTHON_SITE} --trusted-host storage.googleapis.com --trusted-host pypi.python.org  ${SYS}
if [ "${?}" != "0" ] ; then
  echo "Error: installation failed!"
  exit 1
fi

######################################################################################
#
# The following seems to fix the problem of the importer that was unable to find older ProtoBufs:
#
touch ${PACKAGE_LIB_DIR}/google/__init__.py

    # Because we have to provide a fixed name via meta.json ,
    # and $PACKAGE_LIB_DIR depends on the Python version,
    # we solve it by creating a symbolic link with a fixed name.
    #
ln -s $PACKAGE_LIB_DIR ${INSTALL_DIR}/lib
