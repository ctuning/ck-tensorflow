#! /bin/bash

set -e

cd ${INSTALL_DIR}/${PACKAGE_SUB_DIR}

cd "tensorflow/contrib/cmake"
mkdir -p build
cd build

XCMAKE_AR=""
if [ "${CK_AR_PATH_FOR_CMAKE}" != "" ] ; then
    XCMAKE_AR=" -DCMAKE_AR=${CK_AR_PATH_FOR_CMAKE} "
fi

XCMAKE_LD=""
if [ "${CK_LD_PATH_FOR_CMAKE}" != "" ] ; then
    XCMAKE_LD=" -DCMAKE_LINKER=${CK_LD_PATH_FOR_CMAKE} "
fi

echo "========================================================================="
echo "    Running CMake"
echo "========================================================================="

set -o xtrace

${CK_CMAKE} -DCMAKE_BUILD_TYPE=Release \
            ${PACKAGE_CONFIGURE_FLAGS} \
            -DCMAKE_C_COMPILER="${CK_CC_PATH_FOR_CMAKE}" \
            -DCMAKE_C_FLAGS="${CK_CC_FLAGS_FOR_CMAKE}" \
            -DCMAKE_CXX_COMPILER="${CK_CXX_PATH_FOR_CMAKE}" \
            -DCMAKE_CXX_FLAGS="${CK_CXX_FLAGS_FOR_CMAKE}" \
            ${XCMAKE_AR} \
            ${XCMAKE_LD} \
            ${CK_CMAKE_EXTRA} \
            ..

set +o xtrace

echo "========================================================================="
echo "    Running make"
echo "========================================================================="

${CK_MAKE} -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS:-1} tf_python_build_pip_package
