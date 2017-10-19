#! /bin/bash

# CK installation script for TensorFlow package from Makefiles
# http://cKnowledge.org/ai

TENSORFLOW_SRC=${INSTALL_DIR}/src

cd ${TENSORFLOW_SRC}

MAKEFILE_DIR=tensorflow/contrib/makefile
FULL_MAKEFILE_DIR=${TENSORFLOW_SRC}/${MAKEFILE_DIR}

if [ "${CK_ANDROID_ABI}" != "" ] ; then
  export NDK_ROOT=${CK_ANDROID_NDK_ROOT_DIR}
  CK_INCLUDE="-I${CK_ENV_LIB_PROTOBUF_INCLUDE}"
  CK_TF_TARGET="TARGET=ANDROID"
else
  CK_INCLUDE="-I${CK_ENV_LIB_PROTOBUF_HOST_INCLUDE}"
  CK_LIBS="${CK_LIBS} -L${CK_ENV_LIB_PROTOBUF_HOST_LIB}"
  if [ "$TF_MAKE_TARGET" != "" ] ; then
    CK_TF_TARGET="TARGET=${TF_MAKE_TARGET}"
  fi
fi

if [ "${CK_ENV_COMPILER_LLVM_BIN}" != "" ] ; then
  CK_FULL_CXX="${CK_ENV_COMPILER_LLVM_BIN}/${CK_CXX}"
  CK_FULL_AR="${CK_ENV_COMPILER_LLVM_BIN}/${CK_AR}"
  CK_FULL_LD="${CK_ENV_COMPILER_LLVM_BIN}/${CK_LD}"
elif [ "${CK_ENV_COMPILER_GCC_BIN}" != "" ] ; then
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

  patch -p0 < ${ORIGINAL_PACKAGE_DIR}/patch

  ${MAKEFILE_DIR}/download_dependencies.sh
  if [ "${?}" != "0" ]; then
    echo ""
    echo "Error: Downloading dependencies for '${CK_ENV_LIB_TF}/src/${MAKEFILE_DIR}' failed!"
    exit 1
  fi
fi

####################################################################
cp -f ${ORIGINAL_PACKAGE_DIR}/Makefile ${MAKEFILE_DIR}/Makefile

####################################################################
make ${CK_MAKE_OPTS} \
     -f ${MAKEFILE_DIR}/Makefile VERBOSE=1 \
     AR="${CK_FULL_AR}" \
     LD="${CK_FULL_LD}" \
     CXX="${CK_FULL_CXX}" \
     CK_INCLUDE="${CK_INCLUDE}" \
     CK_LIBS="${CK_LIBS}" \
     PROTOC=${CK_ENV_LIB_PROTOBUF_HOST_BIN}/protoc \
     ${CK_TF_TARGET} \
     V=1
if [ "${?}" != "0" ] ; then
  echo ""
  echo "Error: make for android classification failed!"
  exit 1
fi

exit 0
