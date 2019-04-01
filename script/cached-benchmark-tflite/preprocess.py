#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import inspect
import json
import numpy as np
import os
import PIL.Image
import shutil
import sys
import time

CUR_DIR = os.getcwd()

SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(SCRIPT_DIR)

import ck_utils

PYTHONPATH = os.environ['PYTHONPATH']

IMAGE_LIST_FILE = "processed_images_id.json"
PREPROCESSED_FILES = "preprocessed_images_list.txt"
TIMER_JSON = "tmp-ck-timer.json"
ENV_INI = "env.ini"

OPENME = {}

def load_pil_image_into_numpy_array(image, width, height):

  # Check if not RGB and convert to RGB
  if image.mode != 'RGB':
    image = image.convert('RGB')

  image = image.resize((width, height), resample=PIL.Image.BILINEAR)

  # Conver to NumPy array
  img_data = np.array(image.getdata())
  img_data = img_data.astype(np.uint8)

  # Make batch from single image
  batch_shape = (1, height, width, 3)
  batch_data = img_data.reshape(batch_shape)
  return batch_data

def save_preprocessed_image(file_name, image_data):
  image_data.tofile(file_name)

## preprocess(category_index):
def preprocess():
  # Prepare directories
  ck_utils.prepare_dir(PREPROCESS_OUT_DIR)
  ck_utils.prepare_dir(ANNOTATIONS_OUT_DIR)
  ck_utils.prepare_dir(IMAGES_OUT_DIR)
  ck_utils.prepare_dir(DETECTIONS_OUT_DIR)
  ck_utils.prepare_dir(RESULTS_OUT_DIR)

   # Load processing image filenames
  image_files = ck_utils.load_image_list(IMAGES_DIR, IMAGE_COUNT, SKIP_IMAGES)

  # Process images
  load_time_total = 0
  images_processed = 0
  processed_image_ids = []
  preprocessed_list = []
  for file_counter, image_file in enumerate(image_files):
    if FULL_REPORT or (file_counter+1) % 10 == 0:
      print("\nPreprocess image: {} ({} of {})".format(image_file, file_counter+1, len(image_files)))

    # Load image
    load_time_begin = time.time()
    image = PIL.Image.open(os.path.join(IMAGES_DIR, image_file))
    original_width, original_height = image.size
    
    image_id = ck_utils.filename_to_id(image_file, DATASET_TYPE)
    processed_image_ids.append(image_id)

    # The array based representation of the image will be used later 
    # in order to prepare the result image with boxes and labels on it.
    image_data = load_pil_image_into_numpy_array(image, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT)

    # NOTE: Insert additional preprocessing here if needed
    preprocessed_file_name = os.path.join(PREPROCESS_OUT_DIR, image_file)
    save_preprocessed_image(preprocessed_file_name, image_data)
    #preprocessed_list.append([preprocessed_file_name, original_width, original_height])
    preprocessed_list.append([image_file, original_width, original_height])

    load_time = time.time() - load_time_begin
    load_time_total += load_time

    # Exclude first image from averaging
    if file_counter > 0 or IMAGE_COUNT == 1:
      images_processed += 1

  # Save processed images ids list to be able to run
  # evaluation without repeating detections (CK_SKIP_DETECTION=YES)
  with open(IMAGE_LIST_FILE, "w") as f:
    f.write(json.dumps(processed_image_ids))

  with open(PREPROCESSED_FILES, "w") as f:
    for row in preprocessed_list:
      f.write("{};{};{}\n".format(row[0], row[1], row[2]))

  load_avg_time = load_time_total / len(processed_image_ids)

  OPENME["images_load_time_s"] = load_time_total
  OPENME["images_load_time_avg_s"] = load_avg_time

  with open(TIMER_JSON, "w") as o:
    json.dump(OPENME, o, indent=2, sort_keys=True)

  #return processed_image_ids

def ck_preprocess(i):
  def my_env(var): return i['env'].get(var)
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)
  def has_dep_env(dep, var): return var in i['deps'][dep]['dict']['env']

  global PYTHONPATH
  global ANNOTATIONS_OUT_DIR
  global DETECTIONS_OUT_DIR
  global IMAGES_DIR
  global IMAGES_OUT_DIR
  global PREPROCESS_OUT_DIR
  global RESULTS_OUT_DIR

  global DATASET_TYPE
  global FULL_REPORT
  global IMAGE_COUNT
  global MODEL_IMAGE_HEIGHT
  global MODEL_IMAGE_WIDTH
  global SKIP_IMAGES

  print('\n--------------------------------')


  # TF-library path
  if has_dep_env('lib-tensorflow', 'CK_ENV_LIB_TF_LIB'):
    PYTHONPATH = dep_env('lib-tensorflow', 'CK_ENV_LIB_TF_LIB') + ':' + PYTHONPATH

  # TF-model specific value
  if has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_CONVERT_TO_BGR'):
    MODEL_CONVERT_TO_BGR = dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_CONVERT_TO_BGR') == 'YES'
  else:
    MODEL_CONVERT_TO_BGR = False

  # TODO: all weights packages should provide common vars to reveal its 
  # input image size: https://github.com/ctuning/ck-tensorflow/issues/67

  # Model parameters
  if has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH'):
    MODEL_IMAGE_WIDTH = int(dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH'))
    if has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT'):
      MODEL_IMAGE_HEIGHT = int(dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT'))
    else:
      MODEL_IMAGE_HEIGHT = MODEL_IMAGE_WIDTH
  else:
    return {'return': 1, 'error': 'Only TensorFlow model packages are currently supported.'}

  if has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_ROOT') \
     and has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_TFLITE_GRAPH') \
     and has_dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE'):
    MODEL_ROOT = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_ROOT")
    MODEL_TFLITE_GRAPH = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_TFLITE_GRAPH")
    MODEL_TFLITE_GRAPH = os.path.join(MODEL_ROOT, MODEL_TFLITE_GRAPH)
    MODEL_DATASET_TYPE = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE")
    LABELMAP_FILE = dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_LABELMAP_FILE') or ""
    LABELMAP_FILE = os.path.join(MODEL_ROOT, LABELMAP_FILE)
    MODEL_IMAGE_CHANNELS = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_IMAGE_CHANNELS")
    if not MODEL_IMAGE_CHANNELS:
      MODEL_IMAGE_CHANNELS = 3
    else:
      MODEL_IMAGE_CHANNELS = int(MODEL_IMAGE_CHANNELS)
    MODEL_NORMALIZE_DATA = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA") == "YES"
    MODEL_SUBTRACT_MEAN = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_SUBTRACT_MEAN") == "YES"
    MODEL_NEED_BACKGROUND_CORRECTION = dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_NEED_BACKGROUND_CORRECTION") == "YES"
  else:
    print("LABELMAP_FILE = ",dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_LABELMAP_FILE'))
    print("MODEL_ROOT = ", dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_ROOT"))
    print("MODEL_TFLITE_GRAPH = ", dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_TFLITE_GRAPH"))
    print("MODEL_DATASET_TYPE = ", dep_env('weights', "CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE"))

    return {'return': 1, 'error': 'Only TensorFlow model packages are currently supported.'}

  # Dataset parameters
  IMAGES_DIR = dep_env('dataset', "CK_ENV_DATASET_IMAGE_DIR")
  DATASET_TYPE = dep_env('dataset', "CK_ENV_DATASET_TYPE")
  # Annotations can be a directory or a single file, depending on dataset type
  ANNOTATIONS_PATH = dep_env('dataset', "CK_ENV_DATASET_ANNOTATIONS")

  # Program parameters
  NUMBER_OF_PROCESSORS = my_env("CK_CPU_COUNT")
  if not NUMBER_OF_PROCESSORS:
    NUMBER_OF_PROCESSORS = 1
  else:
    NUMBER_OF_PROCESSORS = int(NUMBER_OF_PROCESSORS)
  IMAGE_COUNT = my_env("CK_BATCH_COUNT")
  if not IMAGE_COUNT:
    IMAGE_COUNT = 1
  else:
    IMAGE_COUNT = int(IMAGE_COUNT)
  BATCH_SIZE = my_env("CK_BATCH_SIZE")
  if not BATCH_SIZE:
    BATCH_SIZE = 1
  else:
    BATCH_SIZE = int(BATCH_SIZE)
  SKIP_IMAGES = my_env("CK_SKIP_IMAGES")
  if not SKIP_IMAGES:
    SKIP_IMAGES = 0
  else:
    SKIP_IMAGES = int(SKIP_IMAGES)
  SAVE_IMAGES = my_env("CK_SAVE_IMAGES") == "YES"
  METRIC_TYPE = (my_env("CK_METRIC_TYPE") or DATASET_TYPE).lower()

  ANNOTATIONS_OUT_DIR = os.path.join(CUR_DIR, "annotations")
  DETECTIONS_OUT_DIR = os.path.join(CUR_DIR, "detections")
  IMAGES_OUT_DIR = os.path.join(CUR_DIR, "images")
  PREPROCESS_OUT_DIR = os.path.join(CUR_DIR, "preprocessed")
  RESULTS_OUT_DIR = os.path.join(CUR_DIR, "results")

  FULL_REPORT = my_env("CK_SILENT_MODE") == "NO"
  SKIP_DETECTION = my_env("CK_SKIP_DETECTION") == "YES"
  VERBOSE = my_env("VERBOSE") == "YES"

  # Print settings
  print("Model label map file: " + LABELMAP_FILE)
  print("Model is for dataset: " + MODEL_DATASET_TYPE)

  print("Dataset images: " + IMAGES_DIR)
  print("Dataset annotations: " + ANNOTATIONS_PATH)
  print("Dataset type: " + DATASET_TYPE)

  print("Image count: {}".format(IMAGE_COUNT))
  print("Metric type: " + METRIC_TYPE)
  print("Results directory: {}".format(RESULTS_OUT_DIR))
  print("Temporary annotations directory: " + ANNOTATIONS_OUT_DIR)
  print("Detections directory: " + DETECTIONS_OUT_DIR)
  print("Result images directory: " + IMAGES_OUT_DIR)
  print("Save result images: {}".format(SAVE_IMAGES))
  print("Save preprocessed images: {}".format(PREPROCESSED_FILES))

  # Run detection if needed
  ck_utils.print_header("Process images")
  if SKIP_DETECTION:
    print("\nSkip detection, evaluate previous results")
  else:
    preprocess()

  ENV={}

  ENV["PYTHONPATH"] = PYTHONPATH

  ENV["ANNOTATIONS_PATH"] = ANNOTATIONS_PATH

  ENV["ANNOTATIONS_OUT_DIR"] = ANNOTATIONS_OUT_DIR
  ENV["DETECTIONS_OUT_DIR"] = DETECTIONS_OUT_DIR
  ENV["IMAGES_DIR"] = IMAGES_DIR
  ENV["IMAGES_OUT_DIR"] = IMAGES_OUT_DIR
  ENV["PREPROCESS_OUT_DIR"] = PREPROCESS_OUT_DIR
  ENV["RESULTS_OUT_DIR"] = RESULTS_OUT_DIR

  ENV["IMAGE_LIST_FILE"] = IMAGE_LIST_FILE
  ENV["LABELMAP_FILE"] = LABELMAP_FILE
  ENV["PREPROCESSED_FILES"] = PREPROCESSED_FILES

  ENV["MODEL_DATASET_TYPE"] = MODEL_DATASET_TYPE
  ENV["MODEL_IMAGE_CHANNELS"] = MODEL_IMAGE_CHANNELS
  ENV["MODEL_IMAGE_HEIGHT"] = MODEL_IMAGE_HEIGHT
  ENV["MODEL_IMAGE_WIDTH"] = MODEL_IMAGE_WIDTH
  ENV["MODEL_NEED_BACKGROUND_CORRECTION"] = MODEL_NEED_BACKGROUND_CORRECTION
  ENV["MODEL_NORMALIZE_DATA"] = MODEL_NORMALIZE_DATA
  ENV["MODEL_ROOT"] = MODEL_ROOT
  ENV["MODEL_SUBTRACT_MEAN"] = MODEL_SUBTRACT_MEAN
  ENV["MODEL_TFLITE_GRAPH"] = MODEL_TFLITE_GRAPH

  ENV["DATASET_TYPE"] = DATASET_TYPE
  ENV["METRIC_TYPE"] = METRIC_TYPE

  ENV["BATCH_SIZE"] = BATCH_SIZE
  ENV["IMAGE_COUNT"] = IMAGE_COUNT
  ENV["SAVE_IMAGES"] = SAVE_IMAGES
  ENV["SKIP_IMAGES"] = SKIP_IMAGES
  ENV["SKIP_DETECTION"] = SKIP_DETECTION

  ENV["FULL_REPORT"] = FULL_REPORT
  ENV["NUMBER_OF_PROCESSORS"] = NUMBER_OF_PROCESSORS
  ENV["TIMER_JSON"] = TIMER_JSON
  ENV["VERBOSE"] = VERBOSE

  with open("env.ini", "w") as o:
    for i in ENV:
      o.write('{}={}\n'.format(i,ENV[i]))
  return {
    'return': 0
  }