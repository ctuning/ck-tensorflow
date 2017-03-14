#! /bin/bash
MODEL=$CK_ENV_MODEL_TENSORFLOW/classify_image_graph_def.pb
LABELS=$CK_ENV_MODEL_TENSORFLOW/imagenet_synset_to_human_label_map.txt
IMAGE=$CK_ENV_MODEL_TENSORFLOW/cropped_panda.jpg

if [ ! -z ${CK_TF_MODEL} ]; then
    MODEL=${CK_TF_MODEL}
fi
if [ ! -z ${CK_TF_LABELS} ]; then
    LABELS=${CK_TF_LABELS}
fi
if [ ! -z ${CK_TF_IMAGE} ]; then
    IMAGE=${CK_TF_IMAGE}
fi


cd ${CK_ENV_LIB_TF}/src && bazel-bin/tensorflow/examples/label_image/label_image \
--graph=${MODEL} \
--labels=${LABELS} \
--image=${IMAGE}
