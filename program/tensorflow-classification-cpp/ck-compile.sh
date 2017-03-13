#! /bin/bash
TF_PATH=$CK_ENV_LIB_TF_SRC/tensorflow/examples/label_image

cp ../classification.cpp $TF_PATH/main.cc

cd $CK_ENV_LIB_TF_SRC && bazel build tensorflow/examples/label_image

if [ ! -f $TF_PATH/data/imagenet_comp_graph_label_strings.txt ] || \
[ ! -f $TF_PATH/data/tensorflow_inception_graph.pb ]; then
    echo "\n Downloading default labels and model files \n"
    wget https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip -O \
    $TF_PATH/data/inception_dec_2015.zip \
    && unzip $TF_PATH/data/inception_dec_2015.zip -d $TF_PATH/data
fi