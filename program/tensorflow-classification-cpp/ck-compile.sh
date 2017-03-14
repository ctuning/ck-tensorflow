#! /bin/bash
TF_PATH=${CK_ENV_LIB_TF}/src/tensorflow/examples/label_image

cp ../classification.cpp ${TF_PATH}/main.cc

cd ${CK_ENV_LIB_TF}/src && bazel build tensorflow/examples/label_image
