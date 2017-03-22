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
