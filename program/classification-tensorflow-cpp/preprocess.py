#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import imp
import os
import re
import json
import shutil
import numpy as np
import scipy.io
from scipy.ndimage import zoom

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


def ck_preprocess(i):
  print('\n--------------------------------')
  def my_env(var): return i['env'].get(var)
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)

  # Init variables from environment
  CUR_DIR = os.getcwd()
  MODEL_WEIGHTS = dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_WEIGHTS')
  MODEL_DIR, _ = os.path.split(MODEL_WEIGHTS)
  MODEL_FROZEN_FILE = None # To be assigned
  MODEL_IMAGE_WIDTH = int(dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH"))
  MODEL_IMAGE_HEIGHT = int(dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT"))
  assert MODEL_IMAGE_WIDTH == MODEL_IMAGE_HEIGHT, "Only square images are supported at this time"
  IMAGE_SIZE = MODEL_IMAGE_WIDTH
  MODEL_NORMALIZE_DATA = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA") == "YES"
  MODEL_CONVERT_TO_BGR = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_CONVERT_TO_BGR") == "YES"
  BATCH_COUNT = int(my_env('CK_BATCH_COUNT'))
  BATCH_SIZE = int(my_env('CK_BATCH_SIZE'))
  IMAGES_COUNT = BATCH_COUNT * BATCH_SIZE
  SKIP_IMAGES = int(my_env('CK_SKIP_IMAGES'))
  IMAGE_DIR = dep_env('imagenet-val', 'CK_ENV_DATASET_IMAGENET_VAL')
  IMAGE_FILE = my_env('CK_IMAGE_FILE')
  MODE_SUFFIX = '-{}-{}-{}'.format(IMAGE_SIZE, BATCH_SIZE, BATCH_COUNT)
  RESULTS_DIR = 'predictions'
  IMAGE_LIST_FILE = 'image_list.txt'
  INPUT_LAYER_NAME = 'input'
  OUTPUT_LAYER_NAME = 'output'

  # Preprocessing options:
  #
  # CK_TMP_IMAGE_SIZE - if this set and greater tnan IMAGE_SIZE
  #                     then images will be scaled to this size
  #                     and then cropped to target size
  # CK_CROP_PERCENT   - if TMP_IMAGE_SIZE is not set,
  #                     images will be cropped to this percent
  #                     and then scaled to target size
  #
  TMP_IMAGE_SIZE = int(my_env('CK_TMP_IMAGE_SIZE'))
  CROP_PERCENT = float(my_env('CK_CROP_PERCENT'))
  SUBTRACT_MEAN = my_env("CK_SUBTRACT_MEAN") == "YES"

  # Dir for caching of prepared images
  # Store preprocessed images in sources directory, not in `tmp`, as
  # `tmp` directory can de cleaned between runs and caches will be lost.
  CACHE_DIR_ROOT = os.path.join('..', 'preprocessed')
  CACHE_DIR = os.path.join(CACHE_DIR_ROOT, '{}-{}-{}'.format(IMAGE_SIZE, TMP_IMAGE_SIZE, CROP_PERCENT))
  RECREATE_CACHE = my_env("CK_RECREATE_CACHE") == "YES"

  # Find frozed graph file and graph info file
  for filename in os.listdir(MODEL_DIR):
    if filename.endswith('.pb'):
      MODEL_FROZEN_FILE = os.path.join(MODEL_DIR, filename)
    elif filename.endswith('_info.txt'):
      # Read input and output layer names from graph info file
      with open(os.path.join(MODEL_DIR, filename), 'r') as f:
        for line in f:
          key_name = line.split(' ')
          if len(key_name) == 2:
            if key_name[0] == 'Input:':
              INPUT_LAYER_NAME = key_name[1].strip()
            elif key_name[0] == 'Output:':
              OUTPUT_LAYER_NAME = key_name[1].strip()

  assert MODEL_FROZEN_FILE, "Frozen graph is not found in selected model package"

  # Single file mode
  if IMAGE_FILE:
    image_dir, IMAGE_FILE = os.path.split(IMAGE_FILE)
    # If only filename is set, assume that file is in images package
    if not image_dir:
      image_dir = IMAGE_DIR
    else:
      IMAGE_DIR = image_dir
    assert os.path.isfile(os.path.join(IMAGE_DIR, IMAGE_FILE)), "Input file does not exist"
    IMAGES_COUNT = 1
    BATCH_SIZE = 1
    BATCH_COUNT = 1
    SKIP_IMAGES = 1
    RECREATE_CACHE = True
    CACHE_DIR = os.path.join(CACHE_DIR_ROOT, 'single-image')
    print('Single file mode')
    print('Input image file: {}'.format(IMAGE_FILE))

  print('Input images dir: {}'.format(IMAGE_DIR))
  print('Preprocessed images dir: {}'.format(CACHE_DIR))
  print('Results dir: {}'.format(RESULTS_DIR))
  print('Frozen graph: {}'.format(MODEL_FROZEN_FILE))
  print('Image size: {}x{}'.format(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH))
  print('Input layer: {}'.format(INPUT_LAYER_NAME))
  print('Output layer: {}'.format(OUTPUT_LAYER_NAME))
  print('Batch size: {}'.format(BATCH_SIZE))
  print('Batch count: {}'.format(BATCH_COUNT))
  print('Skip images: {}'.format(SKIP_IMAGES))

  # Prepare cache dir
  if not os.path.isdir(CACHE_DIR_ROOT):
    os.mkdir(CACHE_DIR_ROOT)
  if RECREATE_CACHE:
    if os.path.isdir(CACHE_DIR):
      shutil.rmtree(CACHE_DIR) 
  if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)

  # Prepare results directory
  if os.path.isdir(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
  os.mkdir(RESULTS_DIR)


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
    # Repeat last image to make full last batch
    if len(images) < IMAGES_COUNT:
      for _ in range(IMAGES_COUNT-len(images)):
        images.append(images[-1])
    return images


  # Load processing image filenames and
  if IMAGE_FILE:
    image_list = [IMAGE_FILE]
  else:
    image_list = load_image_list()


  # Returns path to preprocessed image in cache directory
  def get_cached_path(image_file_name):
    return os.path.join(CACHE_DIR, image_file_name)


  # Returns path to source image in dataset directory
  def get_original_path(image_file_name):
    return os.path.join(IMAGE_DIR, image_file_name)


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


  # Preprocess images which are not cached yet
  print('Preprocess images...')
  for image_file in image_list:
    if os.path.isfile(get_cached_path(image_file)):
      continue
    image_data = load_image(get_original_path(image_file))
    image_data.tofile(get_cached_path(image_file))

  # Save list of images to be classified
  with open(IMAGE_LIST_FILE, 'w') as f:
    for image_file in image_list:
      f.write(image_file + '\n')

  # Setup parameters for program
  new_env = {
    'RUN_OPT_IMAGE_DIR': CACHE_DIR,
    'RUN_OPT_IMAGE_LIST': IMAGE_LIST_FILE,
    'RUN_OPT_IMAGE_SIZE': str(IMAGE_SIZE),
    'RUN_OPT_BATCH_SIZE': str(BATCH_SIZE),
    'RUN_OPT_BATCH_COUNT': str(BATCH_COUNT),
    'RUN_OPT_FROZEN_GRAPH': MODEL_FROZEN_FILE,
    'RUN_OPT_RESULT_DIR': RESULTS_DIR,
    'RUN_OPT_NORMALIZE_DATA': str(1 if MODEL_NORMALIZE_DATA else 0),
    'RUN_OPT_SUBTRACT_MEAN': str(1 if SUBTRACT_MEAN else 0),
    'RUN_OPT_INPUT_LAYER_NAME': INPUT_LAYER_NAME,
    'RUN_OPT_OUTPUT_LAYER_NAME': OUTPUT_LAYER_NAME
  }
  
  res = {'return': 0, 'new_env': new_env}

  # Some special preparation to run program on Android device
  if i.get('target_os_dict', {}).get('ck_name2', '') == 'android':
    files_to_push = []
    files_to_pull = []

    # We are in the `tmp` directory now.
    # But when pushing on a device, paths are relative to the program
    # sources directory. So remove leading `../` from cache dir name.
    cache_dir = CACHE_DIR[3:]
    new_env['RUN_OPT_IMAGE_DIR'] = cache_dir

    # Preprocessed images
    for image_file in image_list:
      files_to_push.append(os.path.join(cache_dir, image_file))
      files_to_pull.append(os.path.join(RESULTS_DIR, image_file) + '.txt')

    # Set list of additional files to be copied to Android device.
    # We have to set these files via env variables with full paths 
    # in order to they will be copied into remote program dir without sub-paths.
    new_env['RUN_OPT_IMAGE_LIST_PATH'] = os.path.join(CUR_DIR, IMAGE_LIST_FILE)
    new_env['RUN_OPT_FROZEN_GRAPH_PATH'] = MODEL_FROZEN_FILE
    files_to_push.append('$<<RUN_OPT_IMAGE_LIST_PATH>>$')
    files_to_push.append('$<<RUN_OPT_FROZEN_GRAPH_PATH>>$')

    # On Android graph will be opened from current path, so strip directory
    new_env['RUN_OPT_FROZEN_GRAPH'] = os.path.split(MODEL_FROZEN_FILE)[1]

    res['run_input_files'] = files_to_push
    res['run_output_files'] = files_to_pull

  print('--------------------------------\n')
  return res

