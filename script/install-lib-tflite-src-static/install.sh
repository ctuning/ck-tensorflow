#! /bin/bash

#
# Installation script for the TensorFlow Lite library (TFLite).
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
#
# This script is for installing packages `lib-tflite-*-src-static`.
# It builds a static TFLite library using scripts provided in the
# `${CK_TOOLS}/lib-tflite-*-src-static/src/tensorflow/lite/tools/make/` directory.
#

function exit_if_error() {
 if [ "${?}" != "0" ]; then exit 1; fi
}

function remove_dir_if_exists() {
  if [ -d $1 ]; then rm -rdf $1; fi
}

function exclude_from_build() {
  if [ -f $1 ]; then mv $1 $1~; fi
}


if [ "${PACKAGE_GIT}" == "YES" ] ; then
  echo "--------------------------------";
  echo "Cloning package ${PACKAGE_URL} ..."
  echo ""
  remove_dir_if_exists ${PACKAGE_SUB_DIR}
  git clone ${PACKAGE_GIT_CLONE_FLAGS} ${PACKAGE_URL} ${PACKAGE_SUB_DIR}
  exit_if_error

  echo "--------------------------------";
  echo "Checking out branch ${PACKAGE_GIT_CHECKOUT} ..."
  echo ""
  cd ${PACKAGE_SUB_DIR}
  git checkout ${PACKAGE_GIT_CHECKOUT}
  exit_if_error
fi

cd ${INSTALL_DIR}/${PACKAGE_SUB_DIR}

if [ "${PACKAGE_PATCH}" == "YES" ] ; then
  if [ -d ${ORIGINAL_PACKAGE_DIR}/patch.${CK_TARGET_OS_ID} ] ; then
    echo "--------------------------------";
    echo "Patching source directory ..."
    echo ""
    for i in ${ORIGINAL_PACKAGE_DIR}/patch.${CK_TARGET_OS_ID}/*
    do
      echo "$i"
      patch -p1 < $i
      exit_if_error
    done
  fi
fi

echo "--------------------------------";
echo "Preparing sources ..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLITE_DIR=${INSTALL_DIR}/${PACKAGE_SUB_DIR}/tensorflow/lite
TFLITE_MAKE_DIR=${TFLITE_DIR}/tools/make

remove_dir_if_exists ${TFLITE_MAKE_DIR}/gen

cp ${SCRIPT_DIR}/android_makefile.inc ${TFLITE_MAKE_DIR}/targets/

echo "--------------------------------";
echo "Download dependencies ..."
echo ""

if [ ! -d ${TFLITE_MAKE_DIR}/downloads ]; then
  ${TFLITE_MAKE_DIR}/download_dependencies.sh
  exit_if_error
fi

echo "--------------------------------";
echo "Building ..."
echo ""

if [[ "${CK_ANDROID_NDK_ROOT_DIR}" ]]; then
  echo
  echo "Building Android package..."
  make -f ${TFLITE_MAKE_DIR}/Makefile \
       -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} \
         TARGET=ANDROID \
         ARCH="${HOSTTYPE}" \
         NDK_ROOT="${CK_ANDROID_NDK_ROOT_DIR}" \
         ANDROID_API="${CK_ANDROID_API_LEVEL}" \
         ANDROID_ARCH="${CK_ANDROID_ABI}"
  exit_if_error
else
  echo
  echo "Building Linux package..."

  make -f ${TFLITE_MAKE_DIR}/Makefile \
       -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS}
  exit_if_error
fi

# Copy target files
remove_dir_if_exists ${INSTALL_DIR}/lib
mkdir ${INSTALL_DIR}/lib
cp ${TFLITE_MAKE_DIR}/gen/lib/libtensorflow-lite.a ${INSTALL_DIR}/lib

return 0
