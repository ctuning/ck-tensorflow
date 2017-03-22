#! /bin/bash
TMP_DIR=$(pwd)
PROGRAM_DIR=$(dirname $PWD)

if [ ! -z ${CK_ANDROID_NDK_PLATFORM} ]; then
    #sudo apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev
    export NDK_ROOT=${CK_ANDROID_NDK_ROOT_DIR}

    #compile static library for libjpeg
    if [ ! -f "${CK_LIB_LIBJPEG_TENSORFLOW}/obj/local/armeabi-v7a/libjpeg.a" ]; then
        cd ${CK_LIB_LIBJPEG_TENSORFLOW} && ${NDK_ROOT}/ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android.mk APP_ABI=armeabi-v7a obj/local/armeabi-v7a/libjpeg.a
        if [ "${?}" != "0" ] ; then
            echo ""
            echo "Error: Compiling static library for libjpeg failed!"
            exit 1
        fi
    fi

    cd ${CK_ENV_LIB_TF}/src
    MAKEFILE_DIR=tensorflow/contrib/makefile
    
    if [ ! -d "${MAKEFILE_DIR}/downloads" ]; then
        ${MAKEFILE_DIR}/download_dependencies.sh
        if [ "${?}" != "0" ]; then
            echo ""
            echo "Error: Downloading dependencies for '${CK_ENV_LIB_TF}/src/${MAKEFILE_DIR}' failed!"
            exit 1
        fi
    fi
    
    if [ ! -d "${MAKEFILE_DIR}/gen/protobuf" ]; then
        tensorflow/contrib/makefile/compile_android_protobuf.sh -c
        if [ "${?}" != "0" ] ; then
            echo ""
            echo "Error: Compiling android protobuf for '${CK_ENV_LIB_TF}/src/${MAKEFILE_DIR}' failed!"
            exit 1
        fi
    fi

    cp ${PROGRAM_DIR}/classification.cpp ${MAKEFILE_DIR}/samples/classification.cc
    cp ${PROGRAM_DIR}/Makefile ${MAKEFILE_DIR}/Makefile
    make -f ${MAKEFILE_DIR}/Makefile TARGET=ANDROID
    if [ "${?}" != "0" ] ; then
        echo ""
        echo "Error: make for android classification failed!"
        exit 1
    fi

    cp ${CK_ENV_LIB_TF}/src/${MAKEFILE_DIR}/gen/bin/benchmark ${TMP_DIR}/classification
else
    TF_PATH=${CK_ENV_LIB_TF}/src/tensorflow/examples/label_image
    cp ../classification.cpp ${TF_PATH}/main.cc
    cd ${CK_ENV_LIB_TF}/src && bazel build tensorflow/examples/label_image
    if [ "${?}" != "0" ] ; then
        echo ""
        echo "Error: bazel build for classification failed!"
        exit 1
    fi
    cp ${CK_ENV_LIB_TF}/src/bazel-bin/tensorflow/examples/label_image/label_image ${TMP_DIR}/classification
fi