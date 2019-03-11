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


# Load list of images to be processed
def load_image_list(images_dir, images_count, skip_images):
  assert os.path.isdir(images_dir), 'Input dir does not exit'
  files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
  files = [f for f in files if re.search(r'\.jpg$', f, re.IGNORECASE)
                            or re.search(r'\.jpeg$', f, re.IGNORECASE)]
  assert len(files) > 0, 'Input dir does not contain image files'
  files = sorted(files)[skip_images:]
  assert len(files) > 0, 'Input dir does not contain more files'
  images = files[:images_count]
  # Repeat last image to make full last batch
  if len(images) < images_count:
    for _ in range(images_count-len(images)):
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
def load_image(image_path,            # Full path to processing image
               target_size,           # Desired size of resulting image
               intermediate_size = 0, # Scale to this size then crop to target size
               crop_percentage = 0,   # Crop to this percentage then scale to target size
               convert_to_bgr = False # Swap image channel RGB -> BGR
               ):
  img = scipy.misc.imread(image_path)

  # check if grayscale and convert to RGB
  if len(img.shape) == 2:
      img = np.dstack((img,img,img))

  # drop alpha-channel if present
  if img.shape[2] > 3:
      img = img[:,:,:3]

  # Resize and crop
  if intermediate_size > target_size:
    img = resize_img(img, intermediate_size)
    img = crop_img(img, float(target_size)/float(intermediate_size))
  elif crop_percentage > 0:
    img = crop_img(img, float(crop_percentage)/100.0)
    img = resize_img(img, target_size)

  # Convert to BGR
  if convert_to_bgr:
    swap_img = np.array(img)
    tmp_img = np.array(swap_img)
    tmp_img[:, :, 0] = swap_img[:, :, 2]
    tmp_img[:, :, 2] = swap_img[:, :, 0]
    img = tmp_img

  return img


def ck_preprocess(i):
  print('\n--------------------------------')
  def my_env(var): return i['env'].get(var)
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)
  def has_dep_env(dep, var): return var in i['deps'][dep]['dict']['env']

  # Init variables from environment

  # TF-model specific value
  if has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_CONVERT_TO_BGR'):
    MODEL_CONVERT_TO_BGR = dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_CONVERT_TO_BGR') == 'YES'
  else:
    MODEL_CONVERT_TO_BGR = False

  # TODO: all weights packages should provide common vars to reveal its 
  # input image size: https://github.com/ctuning/ck-tensorflow/issues/67
  if has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH'):
    IMAGE_SIZE = int(dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH'))
  elif has_dep_env('weights', 'CK_ENV_ONNX_MODEL_IMAGE_WIDTH'):
    IMAGE_SIZE = int(dep_env('weights', 'CK_ENV_ONNX_MODEL_IMAGE_WIDTH'))
  else:
    if has_dep_env('weights', 'CK_ENV_MOBILENET_RESOLUTION'):
      IMAGE_SIZE = int(dep_env('weights', 'CK_ENV_MOBILENET_RESOLUTION'))
    else:
      return {'return': 1, 'error': 'Only TensorFlow model packages are currently supported.'}
    
  IMAGE_COUNT = int(my_env('CK_BATCH_COUNT')) * int(my_env('CK_BATCH_SIZE'))
  SKIP_IMAGES = int(my_env('CK_SKIP_IMAGES'))
  IMAGE_DIR = dep_env('imagenet-val', 'CK_ENV_DATASET_IMAGENET_VAL')
  IMAGE_FILE = my_env('CK_IMAGE_FILE')
  RESULTS_DIR = 'predictions'
  IMAGE_LIST_FILE = 'image_list.txt'
  TMP_IMAGE_SIZE = int(my_env('CK_TMP_IMAGE_SIZE'))
  CROP_PERCENT = float(my_env('CK_CROP_PERCENT'))
  SUBTRACT_MEAN = my_env('CK_SUBTRACT_MEAN') == "YES"

  # Full path of dir for caching prepared images.
  # Store preprocessed images in sources directory, not in `tmp`, as
  # `tmp` directory can de cleaned between runs and caches will be lost.
  CACHE_DIR_ROOT = my_env('CK_IMG_CACHE_DIR')
  if not CACHE_DIR_ROOT:
    CACHE_DIR_ROOT = os.path.join('..', 'preprocessed')

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
    SKIP_IMAGES = 1
    RECREATE_CACHE = True
    CACHE_DIR = os.path.join(CACHE_DIR_ROOT, 'single-image')
    print('Single file mode')
    print('Input image file: {}'.format(IMAGE_FILE))
  else:
    RECREATE_CACHE = my_env("CK_RECREATE_CACHE") == "YES"
    CACHE_DIR = os.path.join(CACHE_DIR_ROOT, '{}-{}-{}'.format(IMAGE_SIZE, TMP_IMAGE_SIZE, CROP_PERCENT))

  print('Input images dir: {}'.format(IMAGE_DIR))
  print('Preprocessed images dir: {}'.format(CACHE_DIR))
  print('Results dir: {}'.format(RESULTS_DIR))
  print('Image size: {}'.format(IMAGE_SIZE))
  print('Image count: {}'.format(IMAGE_COUNT))
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

  # Load processing images filenames
  if IMAGE_FILE:
    image_list = [IMAGE_FILE]
  else:
    image_list = load_image_list(IMAGE_DIR, IMAGE_COUNT, SKIP_IMAGES)

  # Preprocess images which are not cached yet
  print('Preprocess images...')
  preprocessed_count = 0
  for image_file in image_list:
    cached_path = os.path.join(CACHE_DIR, image_file)
    if not os.path.isfile(cached_path):
      original_path = os.path.join(IMAGE_DIR, image_file)
      image_data = load_image(image_path = original_path,
                              target_size = IMAGE_SIZE,
                              intermediate_size = TMP_IMAGE_SIZE,
                              crop_percentage = CROP_PERCENT,
                              convert_to_bgr = MODEL_CONVERT_TO_BGR)
      image_data.tofile(cached_path)
      preprocessed_count += 1
      if preprocessed_count % 10 == 0:
        print('  Done {} of {}'.format(preprocessed_count, len(image_list)))
  print('  Done {} of {}'.format(len(image_list), len(image_list)))

  # Save list of images to be classified
  with open(IMAGE_LIST_FILE, 'w') as f:
    for image_file in image_list:
      f.write(image_file + '\n')

  # Setup parameters for program
  new_env = {}
  files_to_push = []
  files_to_pull = []

  # Some special preparation to run program on Android device
  if i.get('target_os_dict', {}).get('ck_name2', '') == 'android':
    # When files will being pushed to Android, current path will be sources path,
    # not `tmp` as during preprocessing. So we have to set `files_to_push` accordingly,
    if CACHE_DIR.startswith('..'):
      CACHE_DIR = CACHE_DIR[3:]

    for image_file in image_list:
      files_to_push.append(os.path.join(CACHE_DIR, image_file))
      files_to_pull.append(os.path.join(RESULTS_DIR, image_file) + '.txt')

    # Set list of additional files to be copied to Android device.
    # We have to set these files via env variables with full paths 
    # in order to they will be copied into remote program dir without sub-paths.
    new_env['RUN_OPT_IMAGE_LIST_PATH'] = os.path.join(os.getcwd(), IMAGE_LIST_FILE)
    files_to_push.append('$<<RUN_OPT_IMAGE_LIST_PATH>>$')

  def to_flag(val):
    return 1 if val and (str(val).upper() == "YES" or int(val) == 1) else 0

  # model-specific variable
  normalize = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA") or dep_env('weights', "CK_ENV_ONNX_MODEL_NORMALIZE_DATA")

  new_env['RUN_OPT_IMAGE_LIST'] = IMAGE_LIST_FILE
  new_env['RUN_OPT_RESULT_DIR'] = RESULTS_DIR
  new_env['RUN_OPT_IMAGE_DIR'] = CACHE_DIR
  new_env['RUN_OPT_IMAGE_SIZE'] = IMAGE_SIZE
  new_env['RUN_OPT_NORMALIZE_DATA'] = to_flag(my_env("CK_NORMALIZE_DATA") or normalize)
  new_env['RUN_OPT_SUBTRACT_MEAN'] = to_flag(my_env("CK_SUBTRACT_MEAN"))
  new_env['RUN_OPT_BATCH_COUNT'] = my_env('CK_BATCH_COUNT')
  new_env['RUN_OPT_BATCH_SIZE'] = my_env('CK_BATCH_SIZE')
  new_env['RUN_OPT_SILENT_MODE'] = to_flag(my_env('CK_SILENT_MODE'))
  print(new_env)

  # Run program specific preprocess script
  preprocess_script = os.path.join(os.getcwd(), '..', 'preprocess-next.py')
  if os.path.isfile(preprocess_script):
    print('--------------------------------')
    print('Running program specific preprocessing script ...')
    module = imp.load_source('preprocess', preprocess_script)
    if hasattr(module, 'ck_preprocess'):
      res = module.ck_preprocess(i)
      if res['return'] > 0: return res
      new_env.update(res.get('new_env', {}))
      files_to_push.extend(res.get('run_input_files', []))
      files_to_pull.extend(res.get('run_output_files', []))

      # Preprocessing can return list of additional files to be copied to Android device.
      # These files are given as full paths, and will be copied near the executable.
      files_to_push_by_path = res.get('files_to_push_by_path', {})
      for key in files_to_push_by_path:
        new_env[key] = files_to_push_by_path[key]
        files_to_push.append('$<<' + key + '>>$')

  print('--------------------------------\n')
  return {
    'return': 0,
    'new_env': new_env,
    'run_input_files': files_to_push,
    'run_output_files': files_to_pull,
  }

