#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
#
# MobilenetNet for TensorFlow
# Implementation mobilenet_v1.py is from
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py   
#

import imp
import os
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.ndimage import zoom
from tensorflow.contrib import slim

MODULE_PATH = os.path.dirname(os.getenv('CK_ENV_TENSORFLOW_MODEL_MODULE'))
RESOLUTION = int(os.getenv('CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION', '224'))
MULTIPLIER = float(os.getenv('CK_ENV_TENSORFLOW_MODEL_MOBILENET_MULTIPLIER', '1.0'))

#-----------------------------------------------------------------------

def load_checkpoints(sess, file_prefix):
    print('Restore checkpoints from {}'.format(file_prefix))
    saver = tf.train.Saver()
    saver.restore(sess, file_prefix)

#-----------------------------------------------------------------------

def load_image(image_path):
    img = scipy.misc.imread(image_path)

    # The same image preprocessing steps are used for MobileNet as for Inseption:
    # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

    # Crop the central region of the image with an area containing 87.5% of the original image.
    new_w = int(img.shape[0] * 0.875)
    new_h = int(img.shape[1] * 0.875)
    offset_w = int((img.shape[0] - new_w)/2)
    offset_h = int((img.shape[1] - new_h)/2)
    img = img[offset_w:new_w+offset_w, offset_h:new_h+offset_h, :]

    # Convert to float and normalize
    img = img.astype(np.float)
    img = img / 255.0

    # Zoom to target size
    zoom_w = float(RESOLUTION)/float(img.shape[0])
    zoom_h = float(RESOLUTION)/float(img.shape[1])
    img = zoom(img, [zoom_w, zoom_h, 1])

    # Shift and scale
    img = img - 0.5
    img = img * 2        

    res = {}
    res['data'] = img
    res['shape'] = img.shape
    return res

#-----------------------------------------------------------------------

def inference(input_image):
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



    
