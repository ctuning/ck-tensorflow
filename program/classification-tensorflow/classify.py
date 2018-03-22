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
import re
import tensorflow as tf
import numpy as np

MODEL_MODULE = os.getenv('CK_ENV_TENSORFLOW_MODEL_MODULE')
MODEL_WEIGHTS = os.getenv('CK_ENV_TENSORFLOW_MODEL_WEIGHTS')
MODEL_WEIGHTS_ARE_CHECKPOINTS = os.getenv('CK_ENV_TENSORFLOW_MODEL_WEIGHTS_ARE_CHECKPOINTS') == 'YES'
BATCH_COUNT = int(os.getenv('CK_BATCH_COUNT', 1))
BATCH_SIZE = int(os.getenv('CK_BATCH_SIZE', 1))
IMAGES_COUNT = BATCH_COUNT * BATCH_SIZE
SKIP_IMAGES = int(os.getenv('CK_SKIP_IMAGES', 0))
IMAGE_DIR = os.getenv('CK_ENV_DATASET_IMAGENET_VAL')
AUX_DIR = os.getenv('CK_ENV_DATASET_IMAGENET_AUX')
VALUES_FILE = os.path.join(AUX_DIR, 'val.txt')
CLASSES_FILE = os.path.join(AUX_DIR, 'synset_words.txt')
CLASSES_LIST = []
VALUES_MAP = {}
TOP1 = 0
TOP5 = 0

def load_ImageNet_classes():
  global CLASSES_LIST
  with open(CLASSES_FILE, 'r') as classes_file:
    CLASSES_LIST = classes_file.read().splitlines()
  
  global VALUES_MAP
  with open(VALUES_FILE, 'r') as values_file:
    for _ in range(SKIP_IMAGES):
      values_file.readline().split()
    for _ in range(IMAGES_COUNT):
      val = values_file.readline().split()
      VALUES_MAP[val[0]] = int(val[1])


def get_class_str(class_index):
  obj_class = CLASSES_LIST[class_index]
  if len(obj_class) > 50:
      obj_class = obj_class[:50] + '...'
  return '(%d) %s' % (class_index, obj_class)


# returns list of pairs (prob, class_index)
def get_top5(all_probs):
  probs_with_classes = []
  for class_index in range(len(all_probs)):
    prob = all_probs[class_index]
    probs_with_classes.append((prob, class_index))
  sorted_probs = sorted(probs_with_classes, key = lambda pair: pair[0], reverse=True)
  return sorted_probs[0:5]


# top5 - list of pairs (prob, class_index)
def print_predictions(top5, img_file):
  class_correct = VALUES_MAP[img_file]
  print('---------------------------------------')
  print('%s - %s' % (img_file, get_class_str(class_correct)))
  for prob, class_index in top5:
    print('%.2f - %s' % (prob, get_class_str(class_index)))
  print('---------------------------------------')


# top5 - list of pairs (prob, class_index)
def check_predictions(top5, img_file):
  class_correct = VALUES_MAP[img_file]
  classes = [c[1] for c in top5]
  is_top1 = class_correct == classes[0]
  is_top5 = class_correct in classes
  if is_top1:
    global TOP1
    TOP1 += 1
  if is_top5:
    global TOP5
    TOP5 += 1
  res = {}
  res['accuracy_top1'] = 'yes' if is_top1 else 'no'
  res['accuracy_top5'] = 'yes' if is_top5 else 'no'
  res['class_correct'] = class_correct
  res['class_topmost'] = classes[0]
  res['file_name'] = img_file
  return res


def load_image_list():
  assert os.path.isdir(IMAGE_DIR), 'Input dir does not exit'
  files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
  files = [f for f in files if re.search(r'\.jpg$', f, re.IGNORECASE)
                            or re.search(r'\.jpeg$', f, re.IGNORECASE)]
  assert len(files) > 0, 'Input dir does not contain image files'
  files = sorted(files)[SKIP_IMAGES:]
  assert len(files) > 0, 'Input dir does not contain more files'
  images = files[:IMAGES_COUNT]
  if len(images) < IMAGES_COUNT:
    for _ in range(IMAGES_COUNT-len(images)):
      images.append(images[-1])
  return images


def main(_):
  print('Model module: ' + MODEL_MODULE)
  print('Model weights: ' + MODEL_WEIGHTS)
  print('Input images dir: ' + IMAGE_DIR)
  print('Batch size: %d' % BATCH_SIZE)
  print('Batch count: %d' % BATCH_COUNT)

  exe_begin_time = time.time()

  # Load model implementation module
  model = imp.load_source('tf_model', MODEL_MODULE)

  # Load processing image filenames
  image_list = load_image_list()

  # Load ImageNet classes info
  load_ImageNet_classes()

  # Prepare TF config options
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.per_process_gpu_memory_fraction = float(os.getenv('CK_TF_GPU_MEMORY_PERCENT', 33)) / 100.0
    
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

  # Load images in model specific way
  # Images should be loaded after weights, because of some models (squeezenet)
  # collect some data from weights and use them to preprocess loaded image
  # Result image must be an object containing 'data' and 'shape' fields
  begin_time = time.time()
  images = []
  for i in range(IMAGES_COUNT):
    img_file = os.path.join(IMAGE_DIR, image_list[i])
    images.append(model.load_image(img_file))
  images_load_time = time.time() - begin_time
  print("Images loaded in %fs" % images_load_time)

  frame_predictions = []
  forward_begin_time = time.time()
  with tf.Graph().as_default(), tf.Session(config=config) as sess:
    # Build net
    begin_time = time.time()
    input_shape = (BATCH_SIZE,) + images[0]['shape']
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

    # Run batched mode
    class_total_time = 0
    image_index = 0
    images_processed = 0
    for batch_index in range(BATCH_COUNT):
      print("\nBatch %d" % (batch_index))

      # Classify batch
      begin_time = time.time()
      batch_data = []
      for _ in range(BATCH_SIZE):
        batch_data.append(images[image_index]['data'])
        image_index += 1
      feed = { input_node: batch_data }
      batch_results = output_node.eval(feed_dict=feed)
      class_time = time.time() - begin_time
      print("Batch classified in %fs" % (class_time))
      
      # Exclude first batch from averaging
      if batch_index > 0 or BATCH_COUNT == 1:
        class_total_time += class_time
        images_processed += BATCH_SIZE

      # Process results
      for index_in_batch in range(BATCH_SIZE):
        all_probs = model.get_image_scores(batch_results, index_in_batch)
        global_index = batch_index * BATCH_SIZE + index_in_batch
        top5 = get_top5(all_probs)
        print_predictions(top5, image_list[global_index])
        res = check_predictions(top5, image_list[global_index])
        frame_predictions.append(res)

  accuracy_top1 = TOP1 / float(IMAGES_COUNT)
  accuracy_top5 = TOP5 / float(IMAGES_COUNT)
  class_avg_time = class_total_time / images_processed
  print('\n')
  print('Average classification time: %fs %s' % (class_avg_time, '(first batch excluded)' if BATCH_COUNT > 1 else ''))
  print('Accuracy top 1: %f (%d of %d)' % (accuracy_top1, TOP1, IMAGES_COUNT))
  print('Accuracy top 5: %f (%d of %d)' % (accuracy_top5, TOP5, IMAGES_COUNT))

  forward_time = time.time() - forward_begin_time
  exe_time = time.time() - exe_begin_time
  print('\n')
  print('All batches time: %fs' % forward_time)
  print('Execution time: %fs' % exe_time)

  # Store benchmark results
  openme = {}
  openme['CK_BATCH_SIZE'] = BATCH_SIZE
  openme['CK_BATCH_COUNT'] = BATCH_COUNT
  openme['CK_MODEL_MODULE'] = MODEL_MODULE
  openme['CK_MODEL_WEIGHTS'] = MODEL_WEIGHTS
  openme['CK_IMAGENET_SYNSET_WORDS_TXT'] = CLASSES_FILE
  openme['CK_IMAGENET_VAL_TXT'] = VALUES_FILE
  openme['accuracy_top1'] = accuracy_top1
  openme['accuracy_top5'] = accuracy_top5
  openme['execution_time'] = exe_time
  openme['time_fw_ms'] = forward_time * 1000
  openme['weights_load_time_s'] = weights_load_time
  openme['images_load_time_s'] = images_load_time
  openme['net_create_time_s'] = net_create_time
  openme['prediction_time_total_s'] = class_total_time
  openme['prediction_time_avg_s'] = class_avg_time
  openme['avg_time_ms'] = class_avg_time * 1000
  openme['avg_fps'] = 1.0 / class_avg_time
  openme['total_time_ms'] = class_avg_time * 1000 * BATCH_SIZE
  openme['frame_predictions'] = frame_predictions
  openme['batch_size'] = BATCH_SIZE
  with open('tmp-ck-timer.json', 'w') as o:
    json.dump(openme, o, indent=2, sort_keys=True)

if __name__ == '__main__':
  tf.app.run()
