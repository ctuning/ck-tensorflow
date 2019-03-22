#!/usr/bin/env python3


import os
import numpy as np
from PIL import Image
import tensorflow as tf


model_path          = os.environ['CK_ENV_TENSORFLOW_MODEL_TF_FROZEN_FILEPATH']
input_layer_name    = os.environ['CK_ENV_TENSORFLOW_MODEL_INPUT_LAYER_NAME']
output_layer_name   = os.environ['CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME']
normalize_data      = os.environ['CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA']
imagenet_path       = os.environ['CK_ENV_DATASET_IMAGENET_VAL']
labels_path         = os.environ['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']
data_layout         = os.environ['CK_MODEL_DATA_LAYOUT']

normalize_data_bool = normalize_data in ('YES', 'yes', 'ON', 'on', '1')


def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


def load_and_resize_image(image_filepath, height, width):
    pillow_img = Image.open(image_filepath).resize((width, height)) # sic! The order of dimensions in resize is (W,H)

    input_data = np.float32(pillow_img)

    if normalize_data_bool:
        input_data = input_data/127.5 - 1.0

#    print(np.array(pillow_img).shape)
    nhwc_data = np.expand_dims(input_data, axis=0)

    if data_layout == 'NHWC':
        # print(nhwc_data.shape)
        return nhwc_data
    else:
        nchw_data = nhwc_data.transpose(0,3,1,2)
        # print(nchw_data.shape)
        return nchw_data


def load_a_batch(batch_filenames):
    unconcatenated_batch_data = []
    for image_filename in batch_filenames:
        image_filepath = imagenet_path + '/' + image_filename
        nchw_data = load_and_resize_image( image_filepath, height, width )
        unconcatenated_batch_data.append( nchw_data )
    batch_data = np.concatenate(unconcatenated_batch_data, axis=0)

    return batch_data


def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import the graph_def into a new Graph and return it
    with tf.Graph().as_default() as graph:
        # The value of name variable will prefix every op/node name. The default is "import".
        # Since we don't want any prefix, we have to override it with an empty string.
        tf.import_graph_def(graph_def, name="")

    return graph


labels = load_labels(labels_path)

# Prepare TF config options
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = float(os.getenv('CK_TF_GPU_MEMORY_PERCENT', 33)) / 100.0
num_processors = int(os.getenv('CK_TF_CPU_NUM_OF_PROCESSORS', 0))
if num_processors > 0:
    config.device_count["CPU"] = num_processors


graph = load_graph(model_path)
input_layer = graph.get_tensor_by_name(input_layer_name+':0')
output_layer = graph.get_tensor_by_name(output_layer_name+':0')

model_input_shape = input_layer.shape

if data_layout == 'NHWC':
    (samples, height, width, channels) = model_input_shape
else:
    (samples, channels, height, width) = model_input_shape

print("Data layout: {}".format(data_layout) )
print("Input layer: {}".format(input_layer) )
print("Output layer: {}".format(output_layer) )
print("Expected input shape: {}".format(model_input_shape) )
print("Data normalization: {}".format(normalize_data_bool) )
print("")

starting_index = 1
batch_size = 5
batch_count = 2

with tf.Session(graph=graph, config=config) as sess:

    for batch_idx in range(batch_count):
        print("Batch {}/{}:".format(batch_idx+1,batch_count))
        batch_filenames = [ "ILSVRC2012_val_00000{:03d}.JPEG".format(starting_index + batch_idx*batch_size + i) for i in range(batch_size) ]

        batch_data = load_a_batch( batch_filenames )
        #print(batch_data.shape)

        batch_predictions = sess.run(output_layer, feed_dict={ input_layer: batch_data } )

        for in_batch_idx in range(batch_size):
            softmax_vector = batch_predictions[in_batch_idx]
            top5_indices = list(reversed(softmax_vector.argsort()))[:5]
            print(batch_filenames[in_batch_idx] + ' :')
            for class_idx in top5_indices:
                print("\t{}\t{}".format(softmax_vector[class_idx], labels[class_idx]))
            print("")

