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

tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_file', './data/sample.png', """Input image to be detected.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")

COLORS = {
    'car': (255, 191, 0),
    'cyclist': (0, 191, 255),
    'pedestrian':(255, 0, 191)
}

RUN_COUNT = 5

def make_model(config_func, model_class):
  gpu_id = 0
  mc = config_func()
  mc.BATCH_SIZE = 1
  # model parameters will be restored from checkpoint
  mc.LOAD_PRETRAINED_MODEL = False
  model = model_class(mc, gpu_id)
  return mc, model


def main(_):
  config = tf.ConfigProto(allow_soft_placement=True)
  
  with tf.Graph().as_default(), tf.Session(config=config) as sess:
    # Make model config and implementation
    begin_time = time.time()
    if FLAGS.demo_net == 'squeezeDet':
      mc, model = make_model(kitti_squeezeDet_config, SqueezeDet)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc, model = make_model(kitti_squeezeDetPlus_config, SqueezeDetPlus)
    elif FLAGS.demo_net == 'resnet50':
      mc, model = make_model(kitti_res50_config, ResNet50ConvDet)
    elif FLAGS.demo_net == 'vgg16':
      mc, model = make_model(kitti_vgg16_config, VGG16ConvDet)
    else:
      print('Selected neural net architecture is not supported: %s' % FLAGS.demo_net)
      exit(1)
    print('Model created in %fs' % (time.time() - begin_time))

    # Restore model parameters
    begin_time = time.time()
    saver = tf.train.Saver(model.model_params)
    saver.restore(sess, FLAGS.checkpoint)
    print('Params restored in %fs' % (time.time() - begin_time))

    # Load and preprocess image
    im = cv2.imread(FLAGS.input_file)
    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
    input_image = im - mc.BGR_MEANS

    # Detect image
    total_time = 0
    for i in range(RUN_COUNT):
      begin_time = time.time()
      det_boxes, det_probs, det_class = sess.run(
        [model.det_boxes, model.det_probs, model.det_class],
        feed_dict={model.image_input:[input_image]})
      detect_time = time.time() - begin_time
      print('Detected in %fs' % detect_time)
      if i > 0 or RUN_COUNT == 1:
        total_time += detect_time

    avg_time = total_time/(RUN_COUNT-1) if RUN_COUNT > 1 else total_time
    print('Average detection time: %fs' % avg_time);

    # Filter
    final_boxes, final_probs, final_class = model.filter_prediction(
        det_boxes[0], det_probs[0], det_class[0])

    keep_idx    = [idx for idx in range(len(final_probs)) \
                      if final_probs[idx] > mc.PLOT_PROB_THRESH]
    final_boxes = [final_boxes[idx] for idx in keep_idx]
    final_probs = [final_probs[idx] for idx in keep_idx]
    final_class = [final_class[idx] for idx in keep_idx]

    # Draw boxes
    _draw_box(
        im, final_boxes,
        [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
            for idx, prob in zip(final_class, final_probs)],
        cdict=COLORS
    )

    cv2.imwrite('output.png', im)
    
  openme = {}
  openme['detect_time_ave'] = avg_time
  with open('tmp-ck-timer.json', 'w') as o:
    json.dump(openme, o)


if __name__ == '__main__':
    tf.app.run()
