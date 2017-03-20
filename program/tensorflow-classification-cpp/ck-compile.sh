#! /bin/bash
TMP_DIR=$(pwd)
if [ ! -z ${CK_ANDROID_NDK_PLATFORM} ]; then
    #sudo apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev
    export NDK_ROOT=${CK_ANDROID_NDK_ROOT_DIR}
    cd ${CK_ENV_LIB_TF}/src
    MAKEFILE_DIR=tensorflow/contrib/makefile
    if [ ! -d "${MAKEFILE_DIR}/downloads" ]; then
        ${MAKEFILE_DIR}/download_dependencies.sh
    fi
    if [ ! -d "${MAKEFILE_DIR}/gen/protobuf" ]; then
        tensorflow/contrib/makefile/compile_android_protobuf.sh -c
    fi
    make -f ${MAKEFILE_DIR}/Makefile TARGET=ANDROID
    cp ${CK_ENV_LIB_TF}/src/tensorflow/contrib/makefile/gen/bin/benchmark ${TMP_DIR}/classification
else
    TF_PATH=${CK_ENV_LIB_TF}/src/tensorflow/examples/label_image
    cp ../classification.cpp ${TF_PATH}/main.cc
    cd ${CK_ENV_LIB_TF}/src && bazel build tensorflow/examples/label_image
    cp ${CK_ENV_LIB_TF}/src/bazel-bin/tensorflow/examples/label_image/label_image ${TMP_DIR}/classification
fi