#! /bin/bash

#
# Installation script for the TensorFlow Lite library.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
#
# This is the script for installing packages `lib-tensorflow-*-src-static`.
# It builds static TensorFlow library using provided scripts in
# `${CK-TOOLS}/lib-tensorflow-*-src-static/src/tensorflow/contrib/makefile/` directory.
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
TFLITE_DIR=${INSTALL_DIR}/${PACKAGE_SUB_DIR}/tensorflow/contrib/lite

remove_dir_if_exists ${TFLITE_DIR}/gen

cp ${SCRIPT_DIR}/android_makefile.inc ${TFLITE_DIR}

# These files reference to some external functionality that seems no more available:
# $CK-TOOLS/$TF/src/third_party/fft2d can't be downloaded from the address mentioned there.
# Library itself can be built successfully, but linking of an application
# (e.g.: ck compile program:image-classification-tflite) fails with message
# spectrogram.cc:(.text+0x47a): undefined reference to `rdft'
exclude_from_build ${TFLITE_DIR}/kernels/internal/spectrogram.h
exclude_from_build ${TFLITE_DIR}/kernels/internal/spectrogram.cc
exclude_from_build ${TFLITE_DIR}/kernels/audio_spectrogram.cc
#if [ -f ${TFLITE_DIR}/kernels/internal/spectrogram.h ]; then
#  mv ${TFLITE_DIR}/kernels/internal/spectrogram.h ${TFLITE_DIR}/kernels/internal/spectrogram.~h
#fi
#if [ -f ${TFLITE_DIR}/kernels/internal/spectrogram.cc ]; then
# mv ${TFLITE_DIR}/kernels/internal/spectrogram.cc ${TFLITE_DIR}/kernels/internal/spectrogram.~cc
#fi

echo "--------------------------------";
echo "Download dependencies ..."
echo ""

if [ ! -d tensorflow/contrib/lite/downloads ]; then
  ./tensorflow/contrib/lite/download_dependencies.sh
  exit_if_error
fi

echo "--------------------------------";
echo "Building ..."
echo ""

if [[ "${CK_ANDROID_NDK_ROOT_DIR}" ]]; then
  echo
  echo "Building Android package..."
  
  make -f tensorflow/contrib/lite/Makefile \
       -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} \
         TARGET=ANDROID \
         NDK_ROOT="$CK_ANDROID_NDK_ROOT_DIR" \
         ANDROID_ARCH="$CK_ANDROID_ABI" \
         ANDROID_API="$CK_ANDROID_API_LEVEL" \
         CC_PREFIX="$CC_PREFIX"
  exit_if_error
else
  echo
  echo "Building Linux package..."

  make -f tensorflow/contrib/lite/Makefile \
       -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS}
  exit_if_error
fi

# Copy target files
remove_dir_if_exists ${INSTALL_DIR}/lib
mkdir ${INSTALL_DIR}/lib
cp ${INSTALL_DIR}/${PACKAGE_SUB_DIR}/tensorflow/contrib/lite/gen/lib/libtensorflow-lite.a ${INSTALL_DIR}/lib

return 0
