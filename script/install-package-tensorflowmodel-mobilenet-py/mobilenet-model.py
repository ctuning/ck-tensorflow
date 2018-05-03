#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
#
# MobileNets for TensorFlow
# Implementation mobilenet_v1.py is from
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
#

import imp
import os
import sys
import tensorflow as tf
from tensorflow.contrib import slim

MODULE_PATH = os.path.dirname(os.getenv('CK_ENV_TENSORFLOW_MODEL_MODULE'))
RESOLUTION = int(os.getenv('CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION', '224'))
MULTIPLIER = float(os.getenv('CK_ENV_TENSORFLOW_MODEL_MOBILENET_MULTIPLIER', '1.0'))
VERSION = os.getenv('CK_ENV_TENSORFLOW_MODEL_MOBILENET_VERSION', '1')

#-----------------------------------------------------------------------

def load_checkpoints(sess, file_prefix):
    print('Restore checkpoints from {}'.format(file_prefix))
    saver = tf.train.Saver()
    saver.restore(sess, file_prefix)

#-----------------------------------------------------------------------

def inference_v1(input_image):
    # We can't import module as it's out of current dir
    mobilenet_path = os.path.join(MODULE_PATH, 'mobilenet_v1.py')
    mobilenet_v1 = imp.load_source('mobilenet_v1', mobilenet_path)

    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training = False)):
        logits, end_points = mobilenet_v1.mobilenet_v1(input_image,
                                                   num_classes = 1001,
                                                   is_training = False,
                                                   dropout_keep_prob = 1,
                                                   depth_multiplier = MULTIPLIER)

    return end_points['Predictions']

#-----------------------------------------------------------------------

def inference_v2(input_image):
    # Path to additional modules required by mobilenet_v2
    if MODULE_PATH not in sys.path:
      sys.path.append(MODULE_PATH)

    # We can't import module as it's out of current dir
    mobilenet_path = os.path.join(MODULE_PATH, 'mobilenet_v2.py')
    mobilenet_v2 = imp.load_source('mobilenet_v2', mobilenet_path)

    # Inference mode is created by default
    logits, end_points = mobilenet_v2.mobilenet(input_image,
                                                num_classes = 1001,
                                                finegrain_classification_mode = True,
                                                depth_multiplier = MULTIPLIER)

    return end_points['Predictions']

#-----------------------------------------------------------------------

def inference(input_image):
    if VERSION == '1':
        return inference_v1(input_image)

    if VERSION == '2':
        return inference_v2(input_image)

    raise Exception('Unsupported MobileNet version: {}'.format(VERSION))

#-----------------------------------------------------------------------

def get_image_scores(batch_results, image_index):
    '''
    Returns scores for specific image in batch.
    batch_results - results of evaluation of node returned by the 'inference' method of this model.
    image_index - index of image inside of batch feeded into 'inference'.
    '''
    # This model has one additional class compared with ImageNet (background class),
    # so we need to exclude it
    return batch_results[image_index][1:]

#-----------------------------------------------------------------------




