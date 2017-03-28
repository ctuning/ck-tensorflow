#! /bin/bash

# CK installation script for TensorFlow package from Makefiles
# http://cKnowledge.org/ai

TMP_DIR=$(pwd)
PROGRAM_DIR=$(dirname $PWD)
TENSORFLOW_SRC=${CK_ENV_LIB_TF}/src

cd ${TENSORFLOW_SRC}

MAKEFILE_DIR=tensorflow/contrib/makefile
FULL_MAKEFILE_DIR=${TENSORFLOW_SRC}/${MAKEFILE_DIR}

CK_INCLUDE="-I${CK_LIB_LIBJPEG_INCLUDE} -DXOPENME -I${CK_ENV_LIB_RTL_XOPENME_INCLUDE}"
if [ "${CK_ANDROID_ABI}" != "" ] ; then
  export NDK_ROOT=${CK_ANDROID_NDK_ROOT_DIR}
  CK_INCLUDE="$CK_INCLUDE -I${CK_ENV_LIB_PROTOBUF_INCLUDE}"
  CK_TF_TARGET="TARGET=ANDROID"
else
  CK_INCLUDE="$CK_INCLUDE -I${CK_ENV_LIB_PROTOBUF_HOST_INCLUDE} -DTF_VIA_MAKE"
  if [ "$TF_MAKE_TARGET" != "" ] ; then
    CK_TF_TARGET="TARGET=${TF_MAKE_TARGET}"
  fi
fi

if [ "${CK_ENV_COMPILER_GCC_BIN}" != "" ] ; then
  CK_FULL_CXX="${CK_ENV_COMPILER_GCC_BIN}/${CK_CXX}"
  CK_FULL_AR="${CK_ENV_COMPILER_GCC_BIN}/${CK_AR}"
  CK_FULL_LD="${CK_ENV_COMPILER_GCC_BIN}/${CK_LD}"
else
  CK_FULL_CXX=${CK_CXX}
  CK_FULL_AR=${CK_AR}
  CK_FULL_LD=${CK_LD}
fi

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
cp -f ${PROGRAM_DIR}/Makefile ${MAKEFILE_DIR}/Makefile

####################################################################
#make -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} \
make -j 4 \
     -f ${MAKEFILE_DIR}/Makefile VERBOSE=1 \
     AR="${CK_FULL_AR}" \
     LD="${CK_FULL_LD}" \
     CXX="${CK_FULL_CXX}" \
     CK_INCLUDE="${CK_INCLUDE}" \
     CK_LIBS="-ljpeg ${CK_ENV_LIB_RTL_XOPENME_LIB}/${CK_ENV_LIB_RTL_XOPENME_STATIC_NAME}" \
     CK_LDFLAGS="-L${CK_LIB_LIBJPEG_LIB}" \
     PROTOC=${CK_ENV_LIB_PROTOBUF_HOST_BIN}/protoc \
     ${CK_TF_TARGET} \
     V=1
if [ "${?}" != "0" ] ; then
  echo ""
  echo "Error: make for android classification failed!"
  exit 1
fi

cp -f ${CK_ENV_LIB_TF}/src/${MAKEFILE_DIR}/gen/bin/benchmark ${TMP_DIR}/classification
if [ "${?}" != "0" ]; then
  echo ""
  echo "Error: copying binary file failed!"
  exit 1
fi

exit 0
