#
# Copyright (c) 2017 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
#
# This benchmark is based on demo https://github.com/BichenWuUCB/squeezeDet
# and reuses its modules forked into https://github.com/dividiti/squeezeDet
#

import cv2
import time
import json

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint', '', """Path to the model parameter file.""")
tf.app.flags.DEFINE_string('input_file', '', """Input image to be detected.""")
tf.app.flags.DEFINE_string('demo_net', 'squeezeDet', """Neural net architecture.""")
tf.app.flags.DEFINE_integer('batch_size', 5, """Batch size.""")
tf.app.flags.DEFINE_integer('batch_count', 5, """Number of batches to run.""")

COLORS = {
    'car': (255, 191, 0),
    'cyclist': (0, 191, 255),
    'pedestrian':(255, 0, 191)
}

MODEL = {}
MODEL_CONFIG = {}

def make_model(config_func, model_class):
  gpu_id = 0
  global MODEL
  global MODEL_CONFIG
  MODEL_CONFIG = config_func()
  MODEL_CONFIG.BATCH_SIZE = FLAGS.batch_size
  # model parameters will be restored from checkpoint
  MODEL_CONFIG.LOAD_PRETRAINED_MODEL = False
  MODEL = model_class(MODEL_CONFIG, gpu_id)


def write_output(index, im, det_boxes, det_probs, det_class):
  # Filter
  final_boxes, final_probs, final_class = MODEL.filter_prediction(det_boxes, det_probs, det_class)

  keep_idx = [idx for idx in range(len(final_probs)) \
                    if final_probs[idx] > MODEL_CONFIG.PLOT_PROB_THRESH]
  final_boxes = [final_boxes[idx] for idx in keep_idx]
  final_probs = [final_probs[idx] for idx in keep_idx]
  final_class = [final_class[idx] for idx in keep_idx]

  # Draw boxes
  _draw_box(
      im, final_boxes,
      [MODEL_CONFIG.CLASS_NAMES[idx]+': (%.2f)'% prob \
          for idx, prob in zip(final_class, final_probs)],
      cdict=COLORS
  )
  cv2.imwrite('output_%d.png' % index, im)


def main(_):
  print('Net id: ' + FLAGS.demo_net)
  print('Checkpoint file: ' + FLAGS.checkpoint)
  print('Input image: ' + FLAGS.input_file)
  print('Batch size: %d' % FLAGS.batch_size)
  print('Batch count: %d' % FLAGS.batch_count)

  config = tf.ConfigProto(allow_soft_placement=True)
  
  with tf.Graph().as_default(), tf.Session(config=config) as sess:
    # Make model config and implementation
    begin_time = time.time()
    if FLAGS.demo_net == 'squeezeDet':
      make_model(kitti_squeezeDet_config, SqueezeDet)
    elif FLAGS.demo_net == 'squeezeDet+':
      make_model(kitti_squeezeDetPlus_config, SqueezeDetPlus)
    elif FLAGS.demo_net == 'resnet50':
      make_model(kitti_res50_config, ResNet50ConvDet)
    elif FLAGS.demo_net == 'vgg16':
      make_model(kitti_vgg16_config, VGG16ConvDet)
    else:
      print('Selected neural net architecture is not supported: %s' % FLAGS.demo_net)
      exit(1)
    print('Model created in %fs' % (time.time() - begin_time))

    # Restore model parameters
    begin_time = time.time()
    saver = tf.train.Saver(MODEL.model_params)
    saver.restore(sess, FLAGS.checkpoint)
    print('Params restored in %fs' % (time.time() - begin_time))

    # Detect all batches
    total_time = 0
    images_processed = 0
    for batch_index in range(FLAGS.batch_count):
      # Load and preprocess batch images
      input_images = []
      batch_data = []
      for _ in range(FLAGS.batch_size):
        im = cv2.imread(FLAGS.input_file)
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (MODEL_CONFIG.IMAGE_WIDTH, MODEL_CONFIG.IMAGE_HEIGHT))
        input_images.append(im)
        batch_data.append(im - MODEL_CONFIG.BGR_MEANS)

      # Detect batch
      begin_time = time.time()
      feed = {MODEL.image_input: batch_data}
      det_boxes, det_probs, det_class = sess.run(
        [MODEL.det_boxes, MODEL.det_probs, MODEL.det_class], feed_dict=feed)
      detect_time = time.time() - begin_time

      # Process batch results
      print('Batch detected in %fs' % detect_time)
      if batch_index > 0 or FLAGS.batch_count == 1:
        total_time += detect_time
        images_processed += FLAGS.batch_size

      # Write only first batch results
      if batch_index == 0:
        assert len(det_boxes) == len(det_probs) == len(det_class) == FLAGS.batch_size
        for i in range(FLAGS.batch_size):
          write_output(i, input_images[i], det_boxes[i], det_probs[i], det_class[i])

    avg_time = total_time/images_processed
    print('Average time: %fs' % avg_time);
    print('Average FPS: %f' % (1 / avg_time))

  openme = {}
  openme['detect_time_ave'] = avg_time
  with open('tmp-ck-timer.json', 'w') as o:
    json.dump(openme, o)


if __name__ == '__main__':
    tf.app.run()
