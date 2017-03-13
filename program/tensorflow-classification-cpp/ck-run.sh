#! /bin/bash

cd $CK_ENV_LIB_TF_SRC && bazel-bin/tensorflow/examples/label_image/label_image \
--graph=tensorflow/examples/label_image/data/tensorflow_inception_graph.pb \
--labels=tensorflow/examples/label_image/data/imagenet_comp_graph_label_strings.txt
