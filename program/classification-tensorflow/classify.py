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
import shutil 
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.ndimage import zoom

MODEL_MODULE = os.getenv('CK_ENV_TENSORFLOW_MODEL_MODULE')
MODEL_WEIGHTS = os.getenv('CK_ENV_TENSORFLOW_MODEL_WEIGHTS')
MODEL_WEIGHTS_ARE_CHECKPOINTS = os.getenv('CK_ENV_TENSORFLOW_MODEL_WEIGHTS_ARE_CHECKPOINTS') == 'YES'
MODEL_IMAGE_WIDTH = int(os.getenv("CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH"))
MODEL_IMAGE_HEIGHT = int(os.getenv("CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT"))
assert MODEL_IMAGE_WIDTH == MODEL_IMAGE_HEIGHT, "Only square images are supported at this time"
IMAGE_SIZE = MODEL_IMAGE_WIDTH
MODEL_NORMALIZE_DATA = os.getenv("CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA") == "YES"
MODEL_CONVERT_TO_BGR = os.getenv("CK_ENV_TENSORFLOW_MODEL_CONVERT_TO_BGR") == "YES"
MODEL_MEAN_VALUE = np.array([0, 0, 0], dtype=np.float32)
BATCH_COUNT = int(os.getenv('CK_BATCH_COUNT', 1))
BATCH_SIZE = int(os.getenv('CK_BATCH_SIZE', 1))
IMAGES_COUNT = BATCH_COUNT * BATCH_SIZE
SKIP_IMAGES = int(os.getenv('CK_SKIP_IMAGES', 0))
IMAGE_DIR = os.getenv('CK_ENV_DATASET_IMAGENET_VAL')
IMAGE_FILE = os.getenv('CK_IMAGE_FILE')
AUX_DIR = os.getenv('CK_ENV_DATASET_IMAGENET_AUX')
VALUES_FILE = os.path.join(AUX_DIR, 'val.txt')
CLASSES_FILE = os.path.join(AUX_DIR, 'synset_words.txt')
CLASSES_LIST = []
VALUES_MAP = {}
TOP1 = 0
TOP5 = 0

# Try to use cached prepared images if presented.
# Load as normal, prepare and cache, if no cached data is found.
USE_CACHE = os.getenv("CK_CACHE_IMAGES") == "YES"

# Ignore existed cached images and create new ones.
RECREATE_CACHE = os.getenv("CK_RECREATE_CACHE") == "YES"

# Root dir for cached prepared images
CACHE_DIR = os.getenv("CK_CACHE_DIR")

# Preprocessing options:
#
# CK_TMP_IMAGE_SIZE - if this set and greater tnan IMAGE_SIZE
#                     then images will be scaled to this size
#                     and then cropped to target size
# CK_CROP_PERCENT   - if TMP_IMAGE_SIZE is not set,
#                     images will be cropped to this percent
#                     and then scaled to target size
#
TMP_IMAGE_SIZE = int(os.getenv('CK_TMP_IMAGE_SIZE'))
CROP_PERCENT = float(os.getenv('CK_CROP_PERCENT'))
SUBTRACT_MEAN = os.getenv("CK_SUBTRACT_MEAN") == "YES"
USE_MODEL_MEAN = os.getenv("CK_USE_MODEL_MEAN") == "YES"


# Returns dir for cached prepared images files.
# Dir name consist of values of preparation parameters.
def get_cache_dir():
  return os.path.join(CACHE_DIR, '{}-{}'.format(TMP_IMAGE_SIZE, CROP_PERCENT))


# Returns path to preprocessed image in cache directory
def get_cached_path(image_file_name):
  return os.path.join(get_cache_dir(), image_file_name) + '.npy'


# Returns path to source image in dataset directory
def get_original_path(image_file_name):
  return os.path.join(IMAGE_DIR, image_file_name)


def load_ImageNet_classes():
  global CLASSES_LIST
  with open(CLASSES_FILE, 'r') as classes_file:
    CLASSES_LIST = classes_file.read().splitlines()
  
  global VALUES_MAP
  with open(VALUES_FILE, 'r') as values_file:
    if IMAGE_FILE:
      # Single file mode: try to find this file in values
      for line in values_file:
        file_name, file_class = line.split()
        if file_name == IMAGE_FILE:
          VALUES_MAP[file_name] = int(file_class)
          break
    else:
      # Directory mode: load only required amount of values
      for _ in range(SKIP_IMAGES):
        values_file.readline()
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
  print('---------------------------------------')
  if img_file in VALUES_MAP:
    class_correct = VALUES_MAP[img_file]
    print('%s - %s' % (img_file, get_class_str(class_correct)))
  else:
    print(img_file)
  for prob, class_index in top5:
    print('%.2f - %s' % (prob, get_class_str(class_index)))
  print('---------------------------------------')


# top5 - list of pairs (prob, class_index)
def check_predictions(top5, img_file):
  if not img_file in VALUES_MAP:
    print('Correctness information is not available')
    return {}
    
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


# Load list of images to be processed
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


# Zoom to target size
def resize_img(img, target_size):
  zoom_w = float(target_size)/float(img.shape[0])
  zoom_h = float(target_size)/float(img.shape[1])
  return zoom(img, [zoom_w, zoom_h, 1])


# Crop the central region of the image
def crop_img(img, crop_percent):
  if crop_percent > 0 and crop_percent < 1.0:
    new_w = int(img.shape[0] * crop_percent)
    new_h = int(img.shape[1] * crop_percent)
    offset_w = int((img.shape[0] - new_w)/2)
    offset_h = int((img.shape[1] - new_h)/2)
    return img[offset_w:new_w+offset_w, offset_h:new_h+offset_h, :]
  else:
    return img


# Load and preprocess image
def load_image(image_path):
  img = scipy.misc.imread(image_path)

  # check if grayscale and convert to RGB
  if len(img.shape) == 2:
      img = np.dstack((img,img,img))

  # drop alpha-channel if present
  if img.shape[2] > 3:
      img = img[:,:,:3]

  # Resize and crop
  if TMP_IMAGE_SIZE > IMAGE_SIZE:
    img = resize_img(img, TMP_IMAGE_SIZE)
    img = crop_img(img, float(IMAGE_SIZE)/float(TMP_IMAGE_SIZE))
  else:
    img = crop_img(img, CROP_PERCENT/100.0)
    img = resize_img(img, IMAGE_SIZE)

  # Convert to BGR
  if MODEL_CONVERT_TO_BGR:
    swap_img = np.array(img)
    tmp_img = np.array(swap_img)
    tmp_img[:, :, 0] = swap_img[:, :, 2]
    tmp_img[:, :, 2] = swap_img[:, :, 0]
    img = tmp_img

  return img


# Final prepartion of image before loading into network
def prepare_img(img):
  # Convert to float
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

  return img


# Load images batch
def load_batch(image_list, image_index):
  batch_data = []
  loaded_images = 0
  for _ in range(BATCH_SIZE):
    img_file = image_list[image_index]
    do_load_image = True

    # In cached mode try to find cached preprocessed image
    if USE_CACHE:
      img_file_path = get_cached_path(img_file)
      if os.path.isfile(img_file_path):
        print 'LOAD CACHE', img_file
        dd = np.load(img_file_path)
        print dd.shape
        img_data = dd
        do_load_image = False

    # Load and preprocess image
    if do_load_image:
      img_data = load_image(get_original_path(img_file))
      if USE_CACHE:
        print 'SAVE CACHE', img_file
        np.save(get_cached_path(img_file), img_data)

    img_data = prepare_img(img_data)
    batch_data.append(img_data)
    image_index += 1
    loaded_images += 1
    if loaded_images % 10 == 0:
      print('Images loaded: %d of %d ...' % (loaded_images, BATCH_SIZE))
  return batch_data, image_index


def main(_):
  global IMAGE_DIR
  global IMAGE_FILE
  global BATCH_SIZE
  global BATCH_COUNT
  global IMAGES_COUNT
  global SKIP_IMAGES
  global USE_CACHE
  print('Model module: ' + MODEL_MODULE)
  print('Model weights: ' + MODEL_WEIGHTS)
  print('Image size: {}x{}'.format(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH))
  if IMAGE_FILE:
    print('Single file mode')
    print('Input image file: ' + IMAGE_FILE)
    assert os.path.isfile(IMAGE_FILE)
    IMAGE_DIR, IMAGE_FILE = os.path.split(IMAGE_FILE)
    BATCH_SIZE = 1
    BATCH_COUNT = 1
    IMAGES_COUNT = 1
    SKIP_IMAGES = 0
    USE_CACHE = False
  print('Input images dir: ' + IMAGE_DIR)
  print('Batch size: %d' % BATCH_SIZE)
  print('Batch count: %d' % BATCH_COUNT)

  # Prepare cache dirs
  if USE_CACHE:
    if not os.path.isdir(CACHE_DIR):
      os.mkdir(CACHE_DIR)
    cache_dir = get_cache_dir()
    if RECREATE_CACHE:
      if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir) 
    if not os.path.isdir(cache_dir):
      os.mkdir(cache_dir)

  exe_begin_time = time.time()

  # Load model implementation module
  model = imp.load_source('tf_model', MODEL_MODULE)

  # Load mean value from model is presented
  if hasattr(model, 'get_mean_value'):
    global MODEL_MEAN_VALUE
    MODEL_MEAN_VALUE = model.get_mean_value()
  else:
    global USE_MODEL_MEAN
    USE_MODEL_MEAN = False

  # Load processing image filenames
  if IMAGE_FILE:
    image_list = [IMAGE_FILE]
  else:
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
    
  frame_predictions = []
  forward_begin_time = time.time()
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

    # Run batched mode
    images_load_time = 0
    class_total_time = 0
    image_index = 0
    images_processed = 0
    for batch_index in range(BATCH_COUNT):
      print("\nBatch %d" % (batch_index))
      
      begin_time = time.time()
      batch_data, image_index = load_batch(image_list, image_index)
      load_time = time.time() - begin_time
      images_load_time += load_time
      print("Batch loaded in %fs" % (load_time))

      # Classify batch
      begin_time = time.time()
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
  openme['total_time_ms'] = forward_time * 1000
  openme['weights_load_time_s'] = weights_load_time
  openme['images_load_time_s'] = images_load_time
  openme['net_create_time_s'] = net_create_time
  openme['prediction_time_total_s'] = class_total_time
  openme['prediction_time_avg_s'] = class_avg_time
  openme['avg_time_ms'] = class_avg_time * 1000
  openme['avg_fps'] = 1.0 / class_avg_time
  openme['batch_time_ms'] = class_avg_time * 1000 * BATCH_SIZE
  openme['frame_predictions'] = frame_predictions
  openme['batch_size'] = BATCH_SIZE
  with open('tmp-ck-timer.json', 'w') as o:
    json.dump(openme, o, indent=2, sort_keys=True)

if __name__ == '__main__':
  tf.app.run()
