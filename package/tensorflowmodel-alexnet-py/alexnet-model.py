#
# Copyright (c) 2017-2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
#
# AlexNet for TensorFlow
# Implementation is based on https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#

import tensorflow as tf
import numpy as np

# Pretrained weights
WEIGHTS = {}

# Number of classification classes
# 1000 for ImageNet by default but can be overriden in training
NUM_CLASSES = 1000

# Model mode
# Empty by default, then pretrained weights are loaded from some data file by load_weights()
# Can also be TRAIN or TEST, in these cases weights and biases are variables and can be
# restored from training checkpoins
MODE = ''

# Training params
TRAIN_LAYER_NAMES = []
TRAIN_DROPOUT_RATE = None

#-----------------------------------------------------------------------

def model_name():
    return 'alexnet'

#-----------------------------------------------------------------------

def load_weights(data_path):
    global WEIGHTS
    WEIGHTS = np.load(data_path, encoding='bytes').item()

#-----------------------------------------------------------------------

def get_weight_biases(layer_name, w_shape, b_shape):
    if MODE in ['TRAIN', 'TEST']:
      # When train or test all weights and biases should not be consts
      # but variables, to be able to save them into checkpoint files
      trainable = layer_name in TRAIN_LAYER_NAMES
      with tf.variable_scope(layer_name) as scope:
        weights = tf.get_variable('weights', shape = w_shape, trainable = trainable)
        biases = tf.get_variable('biases', shape = b_shape, trainable = trainable)
    else:
      # In 'normal' mode weights and biases can be consts
      # and graph can be easily freezed into pb-file
      weights = tf.constant(WEIGHTS[layer_name][0])   
      biases = tf.constant(WEIGHTS[layer_name][1])
    return weights, biases

#-----------------------------------------------------------------------

def load_pretrained_weights(session):
  for op_name in WEIGHTS:
    if op_name not in TRAIN_LAYER_NAMES:
      weights_data = WEIGHTS[op_name][0]
      biases_data = WEIGHTS[op_name][1]
      with tf.variable_scope(op_name, reuse = True):
        session.run(tf.get_variable('weights').assign(weights_data))
        session.run(tf.get_variable('biases').assign(biases_data))
    
#-----------------------------------------------------------------------

def build_net(input_image):
    x = tf.cast(input_image, tf.float32)

    x = _conv_layer("conv1", x, filters=(11, 11, 96), stride=(4, 4), group=1)
    x = _lrn_layer('norm1', x)
    x = _pooling_layer('pool1', x)
    x = _conv_layer("conv2", x, filters=(5, 5, 256), group=2)
    x = _lrn_layer('norm2', x)
    x = _pooling_layer('pool2', x)
    x = _conv_layer("conv3", x, filters=(3, 3, 384), group=1)
    x = _conv_layer("conv4", x, filters=(3, 3, 384), group=2)
    x = _conv_layer("conv5", x, filters=(3, 3, 256), group=2)
    x = _pooling_layer('pool5', x)
    x = tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))])
    x = _fc_layer('fc6', x, num_in=6*6*256, num_out=4096)

    if MODE == 'TRAIN':
      x = tf.nn.dropout(x, TRAIN_DROPOUT_RATE)
    
    x = _fc_layer('fc7', x, num_in=4096, num_out=4096)

    if MODE == 'TRAIN':
      x = tf.nn.dropout(x, TRAIN_DROPOUT_RATE)

    x = _fc_layer('fc8', x, num_in=4096, num_out=NUM_CLASSES, relu=False)

    return x

#-----------------------------------------------------------------------

def inference(input_image):
    x = build_net(input_image)
    prob = tf.nn.softmax(x, name='prob')
    return prob

#-----------------------------------------------------------------------

def get_image_scores(batch_results, image_index):
    '''
    Returns scores for specific image in batch.
    batch_results - results of evaluation of node returned by the 'inference' method of this model.
    image_index - index of image inside of batch feeded into 'inference'.
    '''
    return batch_results[image_index]

#-----------------------------------------------------------------------

def _conv_layer(layer_name, input, filters, stride=(1, 1), group=1):
    input_channels = int(input.get_shape()[-1])
    assert input_channels % group == 0
    w_shape = [filters[0], filters[1], input_channels/group, filters[2]]
    b_shape = [filters[2]]
    kernel, biases = get_weight_biases(layer_name, w_shape, b_shape)
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, stride[0], stride[1], 1], padding="SAME", name="layer_name")
    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    x = tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
    return tf.nn.relu(x)

#-----------------------------------------------------------------------

def _pooling_layer(layer_name, input):
    return tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=layer_name)

#-----------------------------------------------------------------------

def _lrn_layer(layer_name, input):
    return tf.nn.local_response_normalization(input, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name=layer_name)

#-----------------------------------------------------------------------

def _fc_layer(layer_name, input, num_in, num_out, relu=True):
    w_shape = [num_in, num_out]
    b_shape = [num_out]
    weights, biases = get_weight_biases(layer_name, w_shape, b_shape)
    if relu:
      return tf.nn.relu_layer(input, weights, biases, name=layer_name)
    else:
      return tf.nn.xw_plus_b(input, weights, biases, name=layer_name)
    
