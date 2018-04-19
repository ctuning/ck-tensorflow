#
# Copyright (c) 2017-2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
#
# SqueezeNet v1.1 (signature pool 1/3/5)
# Implementation is based on https://github.com/avoroshilov/tf-squeezenet
#

import tensorflow as tf
import numpy as np
import scipy.io

WEIGHTS = {}

#-----------------------------------------------------------------------

def load_weights(data_path):
    global WEIGHTS

    weights_raw = scipy.io.loadmat(data_path)
    
    # Converting to needed type
    dtype = np.float32
    for name in weights_raw:
        WEIGHTS[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            WEIGHTS[name].append( kernels.astype(dtype) )
            WEIGHTS[name].append( bias.astype(dtype) )

#-----------------------------------------------------------------------

def get_mean_value():
  return np.array([104.006, 116.669, 122.679], dtype=np.float32)

#-----------------------------------------------------------------------

def get_weights_biases(layer_name):
    weights, biases = WEIGHTS[layer_name]
    biases = biases.reshape(-1)
    return (weights, biases)

#-----------------------------------------------------------------------
    
def inference(input_image):
    x = tf.cast(input_image, tf.float32)

    ### Feature extractor
    
    # conv1 cluster
    layer_name = 'conv1'
    weights, biases = get_weights_biases(layer_name)
    x = _conv_layer(layer_name + '_conv', x, weights, biases, padding='VALID', stride=(2, 2))
    x = _act_layer(layer_name + '_actv', x)
    x = _pool_layer('pool1_pool', x)

    # fire2 + fire3 clusters
    x = _fire_cluster('fire2', x)
    x = _fire_cluster('fire3', x)
    x = _pool_layer('pool3_pool', x)

    # fire4 + fire5 clusters
    x = _fire_cluster('fire4', x)
    x = _fire_cluster('fire5', x)
    x = _pool_layer('pool5_pool', x)

    # remainder (no pooling)
    x = _fire_cluster('fire6', x)
    x = _fire_cluster('fire7', x)
    x = _fire_cluster('fire8', x)
    x = _fire_cluster('fire9', x)
    
    ### Classifier

    # Dropout [use value of 50% when training]
    x = tf.nn.dropout(x, 1)

    # Fixed global avg pool/softmax classifier:
    # [227, 227, 3] -> 1000 classes
    layer_name = 'conv10'
    weights, biases = get_weights_biases(layer_name)
    x = _conv_layer(layer_name + '_conv', x, weights, biases)
    x = _act_layer(layer_name + '_actv', x)
    
    # Global Average Pooling
    x = tf.nn.avg_pool(x, ksize=(1, 13, 13, 1), strides=(1, 1, 1, 1), padding='VALID')

    x = tf.nn.softmax(x, name='prob')
    return x

#-----------------------------------------------------------------------

def get_image_scores(batch_results, image_index):
    '''
    Returns scores for specific image in batch.
    batch_results - results of evaluation of node returned by the 'inference' method of this model.
    image_index - index of image inside of batch feeded into 'inference'.
    '''
    return batch_results[image_index][0][0]

#-----------------------------------------------------------------------

def _fire_cluster(cluster_name, x):
    # central - squeeze
    layer_name = cluster_name + '/squeeze1x1'
    weights, biases = get_weights_biases(layer_name)
    x = _conv_layer(layer_name + '_conv', x, weights, biases, padding='VALID')
    x = _act_layer(layer_name + '_actv', x)
    
    # left - expand 1x1
    layer_name = cluster_name + '/expand1x1'
    weights, biases = get_weights_biases(layer_name)
    x_l = _conv_layer(layer_name + '_conv', x, weights, biases, padding='VALID')
    x_l = _act_layer(layer_name + '_actv', x_l)

    # right - expand 3x3
    layer_name = cluster_name + '/expand3x3'
    weights, biases = get_weights_biases(layer_name)
    x_r = _conv_layer(layer_name + '_conv', x, weights, biases, padding='SAME')
    x_r = _act_layer(layer_name + '_actv', x_r)
    
    # concatenate expand 1x1 (left) and expand 3x3 (right)
    return tf.concat([x_l, x_r], 3)

#-----------------------------------------------------------------------
        
def _conv_layer(layer_name, input, weights, bias, padding='SAME', stride=(1, 1)):
    conv = tf.nn.conv2d(input,
                        tf.constant(weights),
                        strides=(1, stride[0], stride[1], 1),
                        padding=padding,
                        name=layer_name)
    return tf.nn.bias_add(conv, bias)

#-----------------------------------------------------------------------

def _act_layer(layer_name, input):
    return tf.nn.relu(input, name=layer_name)

#-----------------------------------------------------------------------
    
def _pool_layer(layer_name, input, size=(3, 3), stride=(2, 2), padding='VALID'):
    return tf.nn.max_pool(input,
                          ksize=(1, size[0], size[1], 1),
                          strides=(1, stride[0], stride[1], 1),
                          padding=padding,
                          name=layer_name)

#-----------------------------------------------------------------------
