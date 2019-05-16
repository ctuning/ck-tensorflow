#
# Copyright (c) 2017-2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import imp
import time
import json
import os
import sys
import numpy as np
import tensorflow as tf

MODEL_MODULE = os.getenv('CK_ENV_TENSORFLOW_MODEL_MODULE')
MODEL_WEIGHTS = os.getenv('CK_ENV_TENSORFLOW_MODEL_WEIGHTS')
MODEL_WEIGHTS_ARE_CHECKPOINTS = os.getenv('CK_ENV_TENSORFLOW_MODEL_WEIGHTS_ARE_CHECKPOINTS') == 'YES'
MODEL_NORMALIZE_DATA = os.getenv("CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA") == "YES"
MODEL_MEAN_VALUE = np.array([0, 0, 0], dtype=np.float32) # to be populated
BATCH_COUNT = int(os.getenv('CK_BATCH_COUNT', 1))
BATCH_SIZE = int(os.getenv('CK_BATCH_SIZE', 1))
IMAGE_LIST = os.getenv('RUN_OPT_IMAGE_LIST')
IMAGE_DIR = os.getenv('RUN_OPT_IMAGE_DIR')
RESULT_DIR = os.getenv('RUN_OPT_RESULT_DIR')
SUBTRACT_MEAN = os.getenv("CK_SUBTRACT_MEAN") == "YES"
USE_MODEL_MEAN = os.getenv("CK_USE_MODEL_MEAN") == "YES"
IMAGE_SIZE = int(os.getenv('RUN_OPT_IMAGE_SIZE'))
FULL_REPORT = int(os.getenv('RUN_OPT_SILENT_MODE', '0')) == 0

# Load images batch
def load_batch(image_list, image_index):
  batch_data = []
  for _ in range(BATCH_SIZE):
    img_file = os.path.join(IMAGE_DIR, image_list[image_index])
    img = np.fromfile(img_file, np.uint8)
    img = img.reshape((IMAGE_SIZE, IMAGE_SIZE, 3))
    img = img.astype(np.float)
    # Normalize
    if MODEL_NORMALIZE_DATA:
      img = img / 255.0
      img = img - 0.5
      img = img * 2
    # Subtract mean value
    if SUBTRACT_MEAN:
      if USE_MODEL_MEAN:
        img = img - MODEL_MEAN_VALUE
      else:
        img = img - np.mean(img)
    # Put to batch
    batch_data.append(img)
    image_index += 1
  return batch_data, image_index


def main(_):
  global MODEL_MEAN_VALUE
  global USE_MODEL_MEAN

  # Print settings
  print('Model module: ' + MODEL_MODULE)
  print('Model weights: ' + MODEL_WEIGHTS)
  print('Images dir: ' + IMAGE_DIR)
  print('Image list: ' + IMAGE_LIST)
  print('Image size: {}'.format(IMAGE_SIZE))
  print('Batch size: {}'.format(BATCH_SIZE))
  print('Batch count: {}'.format(BATCH_COUNT))
  print('Result dir: ' + RESULT_DIR)
  print('Normalize: {}'.format(MODEL_NORMALIZE_DATA))
  print('Subtract mean: {}'.format(SUBTRACT_MEAN))
  print('Use model mean: {}'.format(USE_MODEL_MEAN))

  # Load model implementation module
  model = imp.load_source('tf_model', MODEL_MODULE)

  # Load mean value from model is presented
  if hasattr(model, 'get_mean_value'):
    MODEL_MEAN_VALUE = model.get_mean_value()
  else:
    USE_MODEL_MEAN = False

  # Load processing image filenames
  image_list = []
  with open(IMAGE_LIST, 'r') as f:
    for s in f: image_list.append(s.strip())

  # Prepare TF config options
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.per_process_gpu_memory_fraction = float(os.getenv('CK_TF_GPU_MEMORY_PERCENT', 33)) / 100.0
  num_processors = int(os.getenv('CK_TF_CPU_NUM_OF_PROCESSORS', 0))
  if num_processors > 0:
    config.device_count["CPU"] = num_processors
    
  setup_time_begin = time.time()

  # Load weights
  # Model implementation should store weights data somewhere into its
  # internal structure as main test is not interested in weights details.
  # If weights are not checkpoints then they are static data (e.g. numpy array files)
  # and can be loaded preliminary before network will have constructed.
  if not MODEL_WEIGHTS_ARE_CHECKPOINTS:
    begin_time = time.time()
    model.load_weights(MODEL_WEIGHTS)
    weights_load_time = time.time() - begin_time
    print("Weights loaded in %fs" % weights_load_time)
    
  with tf.Graph().as_default(), tf.Session(config=config) as sess:
    # Build net
    begin_time = time.time()
    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    input_node = tf.placeholder(dtype=tf.float32, shape=input_shape, name="input")
    output_node = model.inference(input_node)
    net_create_time = time.time() - begin_time
    print("Net created in %fs" % net_create_time)

    # Load weights
    # If weights are checkpoints they only can be restored after network has been built
    if MODEL_WEIGHTS_ARE_CHECKPOINTS:
      begin_time = time.time()
      model.load_checkpoints(sess, MODEL_WEIGHTS)
      weights_load_time = time.time() - begin_time
      print("Weights loaded in %fs" % weights_load_time)

    setup_time = time.time() - setup_time_begin

    # Run batched mode
    test_time_begin = time.time()
    image_index = 0
    total_load_time = 0
    total_classification_time = 0
    first_classification_time = 0
    images_loaded = 0

    for batch_index in range(BATCH_COUNT):
      batch_number = batch_index+1
      if FULL_REPORT or (batch_number % 10 == 0):
        print("\nBatch {} of {}".format(batch_number, BATCH_COUNT))
      
      begin_time = time.time()
      batch_data, image_index = load_batch(image_list, image_index)
      load_time = time.time() - begin_time
      total_load_time += load_time
      images_loaded += BATCH_SIZE
      if FULL_REPORT:
        print("Batch loaded in %fs" % (load_time))

      # Classify batch
      begin_time = time.time()
      feed = { input_node: batch_data }
      batch_results = output_node.eval(feed_dict=feed)
      classification_time = time.time() - begin_time
      if FULL_REPORT:
        print("Batch classified in %fs" % (classification_time))
      
      total_classification_time += classification_time
      # Remember first batch prediction time
      if batch_index == 0:
        first_classification_time = classification_time

      # Process results
      for index_in_batch in range(BATCH_SIZE):
        all_probs = model.get_image_scores(batch_results, index_in_batch)
        global_index = batch_index * BATCH_SIZE + index_in_batch
        res_file = os.path.join(RESULT_DIR, image_list[global_index])
        with open(res_file + '.txt', 'w') as f:
          for prob in all_probs:
            f.write('{}\n'.format(prob))
            
  test_time = time.time() - test_time_begin

  if BATCH_COUNT > 1:
    avg_classification_time = (total_classification_time - first_classification_time) / (images_loaded - BATCH_SIZE)
  else:
    avg_classification_time = total_classification_time / images_loaded

  avg_load_time = total_load_time / images_loaded

  # Store benchmark results
  openme = {}
  openme['setup_time_s'] = setup_time
  openme['test_time_s'] = test_time
  openme['net_create_time_s'] = net_create_time
  openme['weights_load_time_s'] = weights_load_time
  openme['images_load_time_total_s'] = total_load_time
  openme['images_load_time_avg_s'] = avg_load_time
  openme['prediction_time_total_s'] = total_classification_time
  openme['prediction_time_avg_s'] = avg_classification_time
  openme['avg_time_ms'] = avg_classification_time * 1000
  openme['avg_fps'] = 1.0 / avg_classification_time
  openme['batch_time_ms'] = avg_classification_time * 1000 * BATCH_SIZE
  openme['batch_size'] = BATCH_SIZE
  
  with open('tmp-ck-timer.json', 'w') as o:
    json.dump(openme, o, indent=2, sort_keys=True)

if __name__ == '__main__':
  tf.app.run()
