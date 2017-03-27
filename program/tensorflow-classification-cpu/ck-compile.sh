#! /bin/bash

echo ""

TMP_DIR=$(pwd)
PROGRAM_DIR=$(dirname $PWD)
TENSORFLOW_SRC=${CK_ENV_LIB_TF}/src

cd ${TENSORFLOW_SRC}

if [ "${CK_ANDROID_ABI}" != "" ] ; then
  MAKEFILE_DIR=tensorflow/contrib/makefile
  FULL_MAKEFILE_DIR=${TENSORFLOW_SRC}/${MAKEFILE_DIR}

  export NDK_ROOT=${CK_ANDROID_NDK_ROOT_DIR}

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

  ####################################################################
  cp -f ${PROGRAM_DIR}/classification.cpp ${MAKEFILE_DIR}/samples/classification.cc
  cp -f ${PROGRAM_DIR}/Makefile.android ${MAKEFILE_DIR}/Makefile

  ####################################################################
#  make -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} \
  make -j 4 \
       -f ${MAKEFILE_DIR}/Makefile VERBOSE=1 \
       AR="${CK_ENV_COMPILER_GCC_BIN}/${CK_AR}" \
       LD="${CK_ENV_COMPILER_GCC_BIN}/${CK_LD}" \
       CXX="${CK_ENV_COMPILER_GCC_BIN}/${CK_CXX}" \
       CK_INCLUDE="-I${CK_ENV_LIB_PROTOBUF_INCLUDE} -I${CK_LIB_LIBJPEG_INCLUDE}" \
       CK_LIBS="-ljpeg" \
       CK_LDFLAGS="-L${CK_LIB_LIBJPEG_LIB}" \
       PROTOC=${CK_ENV_LIB_PROTOBUF_HOST_BIN}/protoc \
       TARGET=ANDROID \
       V=1
  if [ "${?}" != "0" ] ; then
    echo ""
    echo "Error: make for android classification failed!"
    exit 1
  fi

  cp -f ${CK_ENV_LIB_TF}/src/${MAKEFILE_DIR}/gen/bin/benchmark ${TMP_DIR}/classification

else
  cp -f ${PROGRAM_DIR}/classification.cpp ${TENSORFLOW_SRC}/tensorflow/examples/label_image/main.cc

#  Should be already done by CK tensorflow cpu package
#  ./configure --prefix=${CK_ENV_LIB_TF}/install
#  if [ "${?}" != "0" ] ; then
#     echo ""
#     echo "Error: configuration for TF failed!"
#     exit 1
#  fi

  bazel build tensorflow/examples/label_image
  if [ "${?}" != "0" ] ; then
     echo ""
     echo "Error: bazel build for classification failed!"
     exit 1
  fi

  cp -f ${TENSORFLOW_SRC}/bazel-bin/tensorflow/examples/label_image/label_image ${TMP_DIR}/classification
fi