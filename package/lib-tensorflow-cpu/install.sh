#! /bin/bash

# CK installation script for TensorFlow package
#
# Developer(s): 
#  * Vladislav Zaborovskiy, vladzab@yandex.ru
#  * Grigori Fursin, dividiti/cTuning foundation
#

# PACKAGE_DIR
# INSTALL_DIR
# TENSORFLOW_URL

export TENSORFLOW_SRC_DIR=${INSTALL_DIR}/src
export TENSORFLOW_PKG_DIR=${INSTALL_DIR}/pkg
export TENSORFLOW_LIB_DIR=${INSTALL_DIR}/lib
export TENSORFLOW_INSTALL_DIR=${INSTALL_DIR}

######################################################################################
echo ""
echo "Removing everything from '${TENSORFLOW_PKG_DIR}' and '${TENSORFLOW_LIB_DIR}' ..."
rm -rf ${TENSORFLOW_SRC_DIR}
rm -rf ${TENSORFLOW_PKG_DIR}
rm -rf ${TENSORFLOW_LIB_DIR}

######################################################################################
echo ""
echo "Cloning TensorFlow from '${TENSORFLOW_URL}' to '${TENSORFLOW_SRC_DIR}' ..."
git clone ${TENSORFLOW_URL}  ${TENSORFLOW_SRC_DIR}
if [ "${?}" != "0" ] ; then
  echo "Error: Cloning TensorFlow from '${TENSORFLOW_URL}' failed!"
  exit 1
fi

if [ "${CK_ANDROID_ABI}" != "" ] ; then
  cd ${TENSORFLOW_SRC_DIR}

  MAKEFILE_DIR=tensorflow/contrib/makefile
  FULL_MAKEFILE_DIR=${TENSORFLOW_SRC_DIR}/${MAKEFILE_DIR}

  ####################################################################
  if [ ! -d "${MAKEFILE_DIR}/downloads" ]; then
    echo ""
    echo "Downloading extra dependencies via TF ..."
    echo ""

    ${MAKEFILE_DIR}/download_dependencies.sh
    if [ "${?}" != "0" ]; then
      echo ""
      echo "Error: Downloading dependencies for '${CK_ENV_LIB_TF}/src/${MAKEFILE_DIR}' failed!"
      exit 1
    fi
  fi

  cp -f ${ORIGINAL_PACKAGE_DIR}/classification.cpp ${MAKEFILE_DIR}/samples/classification.cc
  cp -f ${ORIGINAL_PACKAGE_DIR}/Makefile.android ${MAKEFILE_DIR}/Makefile
  if [ "${?}" != "0" ] ; then
    echo ""
    echo "Error: Makefile copying failed!"
    exit 1
  fi

  ####################################################################
  export NDK_ROOT=${CK_ANDROID_NDK_ROOT_DIR}
  make -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} -f ${MAKEFILE_DIR}/Makefile VERBOSE=1 AR="${CK_ENV_COMPILER_GCC_BIN}/${CK_AR}" LD="${CK_ENV_COMPILER_GCC_BIN}/${CK_LD}" CXX="${CK_ENV_COMPILER_GCC_BIN}/${CK_CXX}" CK_INCLUDE="-I${CK_ENV_LIB_PROTOBUF_INCLUDE}" PROTOC=${CK_ENV_LIB_PROTOBUF_HOST_BIN}/protoc TARGET=ANDROID V=1
  if [ "${?}" != "0" ] ; then
    echo ""
    echo "Error: make for android classification failed!"
    exit 1
  fi

  cp -f ${CK_ENV_LIB_TF}/src/${MAKEFILE_DIR}/gen/bin/benchmark ${TMP_DIR}/classification
  exit 0
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  #hack to make it work on mac os
  cd ${TENSORFLOW_INSTALL_DIR} && mkdir -p lib/tensorflow && echo '' > lib/tensorflow/__init__.py
else
  ######################################################################################
  cd ${TENSORFLOW_SRC_DIR}

  echo ""
  echo "Configuring ..."

  source ${PACKAGE_DIR}/export-variables
  ./configure
  if [ "${?}" != "0" ] ; then
    echo "Error: TensorFlow installation configuration failed!"
    exit 1
  fi

  ######################################################################################
  echo ""
  echo "Preparing pip package ..."

  if [ "$TF_NEED_CUDA" == 0 ]; then
  if [ "$TF_NEED_OPENCL" == 0 ]; then
    bazel build -c opt //tensorflow/tools/pip_package:build_pip_package;
  else 
    bazel build -c opt --config=sycl //tensorflow/tools/pip_package:build_pip_package;
  fi
  else 
    bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package;
  fi

  if [ "${?}" != "0" ] ; then
    echo "Error: Bazel building pip package failed"
    exit 1
  fi

  ######################################################################################
  echo ""
  echo "Building pip package(s) ..."

  bazel-bin/tensorflow/tools/pip_package/build_pip_package ${TENSORFLOW_PKG_DIR}
  if [ "${?}" != "0" ] ; then
    echo "Error: Bazel building pip package failed"
    exit 1
  fi

  ######################################################################################
  echo ""
  echo "Installing pip package(s) ..."

  for pip_package in ${TENSORFLOW_PKG_DIR}/*.whl
  do
      if [ "$CK_PYTHON3" == 0 ]; then
          pip install $pip_package -t $TENSORFLOW_LIB_DIR
          if [ "${?}" != "0" ] ; then
              echo "Error: Bazel building pip package failed"
              exit 1
          fi
      else
          pip3 install $pip_package -t $TENSORFLOW_LIB_DIR
          if [ "${?}" != "0" ] ; then
              echo "Error: Bazel building pip package failed"
              exit 1
          fi
      fi
  done
fi
######################################################################################
#echo ""
#echo "Cleaning up directories ..."

#rm -rf $TENSORFLOW_SRC_DIR
#rm -rf $TENSORFLOW_PKG_DIR

exit 0
