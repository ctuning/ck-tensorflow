#
# Copyright (c) 2017-2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
#
# GoogleNet for TensorFlow
# Implementation is based on https://github.com/Marsan-Ma/imgrec/blob/master/lib/googlenet.py
#

import tensorflow as tf
import numpy as np

WEIGHTS = {}

#-----------------------------------------------------------------------

def load_weights(data_path):
    global WEIGHTS
    WEIGHTS = np.load(data_path, encoding='bytes').item()

#-----------------------------------------------------------------------

def inference(input_image):
    x = tf.cast(input_image, tf.float32)

    x = _conv_layer('conv1_7x7_s2', x, stride=(2,2))
    x = _max_pool_layer('pool1_3x3_s2', x)
    x = _lrn_layer('pool1_norm', x)
    x = _conv_layer('conv2_3x3_reduce', x)
    x = _conv_layer('conv2_3x3', x)
    x = _lrn_layer('conv2_norm2', x)
    x = _max_pool_layer('pool2_3x3_s2', x)
    x = _inception_layer('inception_3a', x)
    x = _inception_layer('inception_3b', x)
    x = _max_pool_layer('pool3_3x3_s2', x)
    x = _inception_layer('inception_4a', x)
    x = _inception_layer('inception_4b', x)
    x = _inception_layer('inception_4c', x)
    x = _inception_layer('inception_4d', x)
    x = _inception_layer('inception_4e', x)
    x = _max_pool_layer('pool4_3x3_s2', x)
    x = _inception_layer('inception_5a', x)
    x = _inception_layer('inception_5b', x)
    x = _avg_pool_layer('pool5_7x7_s1', x, padding='VALID')
    x = tf.nn.dropout(x, 0.4, name='pool5_drop')

    kernel = tf.constant(WEIGHTS['loss3_classifier']['weights'])
    biases = tf.constant(WEIGHTS['loss3_classifier']['biases'])
    x = tf.reshape(x, [-1, 1024])
    x = tf.nn.xw_plus_b(x, kernel, biases, name='loss3_classifier')

    return tf.nn.softmax(x, name='prob')

#-----------------------------------------------------------------------

def get_image_scores(batch_results, image_index):
    '''
    Returns scores for specific image in batch.
    batch_results - results of evaluation of node returned by the 'inference' method of this model.
    image_index - index of image inside of batch feeded into 'inference'.
    '''
    return batch_results[4*image_index]

#-----------------------------------------------------------------------

def _conv_layer(layer_name, input, stride=(1, 1)):
    kernel = tf.constant(WEIGHTS[layer_name]['weights'])
    biases = tf.constant(WEIGHTS[layer_name]['biases'])
    conv = tf.nn.conv2d(input, kernel, [1, stride[0], stride[1], 1], padding="SAME", name=layer_name)
    x = tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
    return tf.nn.relu(x)

#-----------------------------------------------------------------------

def _max_pool_layer(layer_name, input, kernel=(3,3), stride=(2,2), padding='SAME'):
    ksize = [1, kernel[0], kernel[1], 1]
    strides = [1, stride[0], stride[1], 1]
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name=layer_name)

#-----------------------------------------------------------------------

def _avg_pool_layer(layer_name, input, kernel=(7,7), stride=(1,1), padding='SAME'):
    ksize = [1, kernel[0], kernel[1], 1]
    strides = [1, stride[0], stride[1], 1]
    return tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding=padding, name=layer_name)

#-----------------------------------------------------------------------

def _lrn_layer(layer_name, input):
    return tf.nn.local_response_normalization(input, depth_radius=5, alpha=0.0001, beta=0.75, bias=1.0, name=layer_name)

#-----------------------------------------------------------------------

def _inception_layer(layer_name, input):
    t1 = _conv_layer(layer_name+'_1x1', input)

    t2 = _conv_layer(layer_name+'_3x3_reduce', input)
    t2 = _conv_layer(layer_name+'_3x3', t2)

    t3 = _conv_layer(layer_name+'_5x5_reduce', input)
    t3 = _conv_layer(layer_name+'_5x5', t3)

    t4 = _max_pool_layer(layer_name+'_pool', input, stride=(1,1))
    t4 = _conv_layer(layer_name+'_pool_proj', t4)

    return tf.concat([t1, t2, t3, t4], 3, name=layer_name+'_output')

#-----------------------------------------------------------------------
