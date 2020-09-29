#! /bin/bash

#
# CK post installation script for tflite.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#

# Environment variables defined by CK:

# PACKAGE_DIR
# INSTALL_DIR


function exit_if_error() {
  message=${1:-"unknown"}
  if [ "${?}" != "0" ]; then
    echo "Error: ${message}!"
    exit 1
  fi
}


SRC_DIR=${INSTALL_DIR}/src/tensorflow/lite
BUILD_DIR=${INSTALL_DIR}/build
LIB_DIR=${INSTALL_DIR}/lib


mkdir -p ${BUILD_DIR}


# Create the build and install dirs
cd ${INSTALL_DIR}

# Configure the package.
read -d '' CMK_CMD <<EO_CMK_CMD
${CK_ENV_TOOL_CMAKE_BIN}/cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CK_CC_PATH_FOR_CMAKE}" \
  -DCMAKE_C_FLAGS="${CK_CC_FLAGS_FOR_CMAKE} ${EXTRA_FLAGS} ${EXTRA_CXXFLAGS}" \
  -DCMAKE_CXX_COMPILER="${CK_CXX_PATH_FOR_CMAKE}" \
  -DCMAKE_CXX_FLAGS="${CK_CXX_FLAGS_FOR_CMAKE} ${EXTRA_FLAGS} ${EXTRA_CXXFLAGS}" \
  -DCMAKE_AR="${CK_AR_PATH_FOR_CMAKE}" \
  -DCMAKE_RANLIB="${CK_RANLIB_PATH_FOR_CMAKE}" \
  -DCMAKE_LINKER="${CK_LD_PATH_FOR_CMAKE}" \
  -DTFLITE_ENABLE_RUY=${PACKAGE_LIB_RUY} \
  -DTFLITE_ENABLE_XNNPACK=${PACKAGE_LIB_XNNPACK} \
  "${SRC_DIR}"
EO_CMK_CMD

# First, print the EXACT command we are about to run
echo "Configuring the package with 'CMake' ..."
echo ${CMK_CMD}

# Now, run it from the build directory.
cd ${BUILD_DIR} && eval ${CMK_CMD}
exit_if_error "CMake failed"

# Now, run the cmake command to build
cd ${BUILD_DIR} && make -j${CK_HOST_CPU_NUMBER_OF_PROCESSORS}
exit_if_error "CMake build failed"


mkdir -p ${LIB_DIR}
if [ ${PACKAGE_VERSION} == '2.3.90' ]
then
    mv ${BUILD_DIR}/libtensorflowlite.a ${BUILD_DIR}/libtensorflow-lite.a
fi

mv ${BUILD_DIR}/libtensorflow-lite.a ${LIB_DIR}

cd ${LIB_DIR}

${CK_AR_PATH_FOR_CMAKE} x ${BUILD_DIR}/_deps/fft2d-build/libfft2d_fftsg.a
${CK_AR_PATH_FOR_CMAKE} x ${BUILD_DIR}/_deps/ruy-build/libruy.a
${CK_AR_PATH_FOR_CMAKE} x ${BUILD_DIR}/_deps/farmhash-build/libfarmhash.a
${CK_AR_PATH_FOR_CMAKE} x ${BUILD_DIR}/_deps/flatbuffers-build/libflatbuffers.a

if [ ${PACKAGE_LIB_XNNPACK} == 'ON' ]
then
    ${CK_AR_PATH_FOR_CMAKE} x ${CK_ENV_LIB_XNNPACK_LIB}/libclog.a
    ${CK_AR_PATH_FOR_CMAKE} x ${CK_ENV_LIB_XNNPACK_LIB}/libcpuinfo.a
    ${CK_AR_PATH_FOR_CMAKE} x ${CK_ENV_LIB_XNNPACK_LIB}/libpthreadpool.a
    ${CK_AR_PATH_FOR_CMAKE} x ${CK_ENV_LIB_XNNPACK_LIB}/libXNNPACK.a
fi

${CK_AR_PATH_FOR_CMAKE} r libtensorflow-lite.a *.o


return 0

