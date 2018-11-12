#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import sys
import os
import json
import numpy as np
import shutil
import time
import PIL

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

import ck_utils
import converter_results
import converter_annotations
import calc_metrics_kitti
import calc_metrics_coco
import calc_metrics_oid

CUR_DIR = os.getcwd()

# Model parameters
FROZEN_GRAPH = os.getenv("CK_ENV_TENSORFLOW_MODEL_FROZEN_GRAPH")
LABELMAP_FILE = os.getenv("CK_ENV_TENSORFLOW_MODEL_LABELMAP_FILE")
MODEL_DATASET_TYPE = os.getenv("CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE")

# Dataset parameters
IMAGES_DIR = os.getenv("CK_ENV_DATASET_IMAGE_DIR")
DATASET_TYPE = os.getenv("CK_ENV_DATASET_TYPE")
# Annotations can be a directory or a single file, depending on dataset type
ANNOTATIONS_PATH = os.getenv("CK_ENV_DATASET_ANNOTATIONS")
ANNOTATIONS_DIR = os.getenv("CK_ENV_DATASET_ANNOTATIONS_DIR")

# Program parameters
IMAGE_COUNT = int(os.getenv('CK_BATCH_COUNT', 1))
SKIP_IMAGES = int(os.getenv('CK_SKIP_IMAGES', 0))
SAVE_IMAGES = os.getenv("CK_SAVE_IMAGES") == "YES"
METRIC_TYPE = (os.getenv("CK_METRIC_TYPE") or DATASET_TYPE).lower()
IMAGES_OUT_DIR = os.path.join(CUR_DIR, "images")
DETECTIONS_OUT_DIR = os.path.join(CUR_DIR, "detections")
ANNOTATIONS_OUT_DIR = os.path.join(CUR_DIR, "annotations")
RESULTS_OUT_DIR = os.path.join(CUR_DIR, "results")
FULL_REPORT = os.getenv('CK_SILENT_MODE') == 'NO'
SKIP_DETECTION = os.getenv('CK_SKIP_DETECTION') == 'YES'
IMAGE_LIST_FILE = 'processed_images_id.json'

OPENME = {}

def make_tf_config():
  mem_percent = float(os.getenv('CK_TF_GPU_MEMORY_PERCENT', 33))
  num_processors = int(os.getenv('CK_TF_CPU_NUM_OF_PROCESSORS', 0))

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.per_process_gpu_memory_fraction = mem_percent / 100.0
  if num_processors > 0:
    config.device_count["CPU"] = num_processors
  return config


def load_pil_image_into_numpy_array(image):
  # Check if not RGB and convert to RGB
  if image.mode != 'RGB':
    image = image.convert('RGB')

  # Conver to NumPy array
  img_data = np.array(image.getdata())
  img_data = img_data.astype(np.uint8)

  # Make batch from single image
  (im_width, im_height) = image.size
  batch_shape = (1, im_height, im_width, 3)
  batch_data = img_data.reshape(batch_shape)
  return batch_data


def get_handles_to_tensors():
  graph = tf.get_default_graph()
  ops = graph.get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  key_list = [
    'num_detections',
    'detection_boxes',
    'detection_scores',
    'detection_classes'
  ]
  for key in key_list:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
  image_tensor = graph.get_tensor_by_name('image_tensor:0')
  return tensor_dict, image_tensor


def save_detection_txt(image_file, image_size, output_dict, category_index):
  (im_width, im_height) = image_size
  file_name = os.path.splitext(image_file)[0]
  res_file = os.path.join(DETECTIONS_OUT_DIR, file_name) + '.txt'
  with open(res_file, 'w') as f:
    f.write('{:d} {:d}\n'.format(im_width, im_height))
    for i in range(output_dict['num_detections']):
      class_id = output_dict['detection_classes'][i]
      if 'display_name' in category_index[class_id]:
        class_name = category_index[class_id]['display_name']
      else:
        class_name = category_index[class_id]['name']
      y1, x1, y2, x2 = output_dict['detection_boxes'][i]
      score = output_dict['detection_scores'][i]
      f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:d} {}\n'\
        .format(x1*im_width, y1*im_height, x2*im_width, y2*im_height, score, class_id, class_name))


def save_detection_img(image_file, image_np, output_dict, category_index):
  if not SAVE_IMAGES: return

  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      line_thickness=2)
  image = PIL.Image.fromarray(image_np)
  image.save(os.path.join(IMAGES_OUT_DIR, image_file))


def detect(category_index):
  # Prepare TF config options
  tf_config = make_tf_config()

  # Prepare directories
  ck_utils.prepare_dir(RESULTS_OUT_DIR)
  ck_utils.prepare_dir(ANNOTATIONS_OUT_DIR)
  ck_utils.prepare_dir(IMAGES_OUT_DIR)
  ck_utils.prepare_dir(DETECTIONS_OUT_DIR)

  # Load processing image filenames
  image_files = ck_utils.load_image_list(IMAGES_DIR, IMAGE_COUNT, SKIP_IMAGES)

  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    setup_time_begin = time.time()
    
    # Make TF graph def from frozen graph file
    begin_time = time.time()
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_GRAPH, 'rb') as f:
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name='')
    graph_load_time = time.time() - begin_time
    print('Graph loaded in {:.4f}s'.format(graph_load_time))

    # NOTE: Load checkpoint here when they are needed

    # Get handles to input and output tensors
    tensor_dict, input_tensor = get_handles_to_tensors()

    setup_time = time.time() - setup_time_begin

    # Process images
    # TODO: implement batched mode
    test_time_begin = time.time()
    image_index = 0
    load_time_total = 0
    detect_time_total = 0
    images_processed = 0
    processed_image_ids = []
    for file_counter, image_file in enumerate(image_files):
      if FULL_REPORT or (file_counter+1) % 10 == 0:
        print('\nDetect image: {} ({} of {})'.format(image_file, file_counter+1, len(image_files)))

      # Load image
      load_time_begin = time.time()
      image = PIL.Image.open(os.path.join(IMAGES_DIR, image_file))
      image_id = ck_utils.filename_to_id(image_file, DATASET_TYPE)
      processed_image_ids.append(image_id)

      # The array based representation of the image will be used later 
      # in order to prepare the result image with boxes and labels on it.
      image_data = load_pil_image_into_numpy_array(image)
      load_time = time.time() - load_time_begin
      load_time_total += load_time
      
      # Detect image
      detect_time_begin = time.time()
      feed_dict = {input_tensor: image_data}
      output_dict = sess.run(tensor_dict, feed_dict)
      detect_time = time.time() - detect_time_begin

      # Exclude first image from averaging
      if file_counter > 0 or IMAGE_COUNT == 1:
        detect_time_total += detect_time
        images_processed += 1

      # Process results
      # All outputs are float32 numpy arrays, so convert types as appropriate
      # TODO: implement batched mode (0 here is the image index in the batch)
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]

      save_detection_txt(image_file, image.size, output_dict, category_index)
      save_detection_img(image_file, image_data[0], output_dict, category_index)

      if FULL_REPORT:
        print('Detected in {:.4f}s'.format(detect_time))

  # Save processed images ids list to be able to run
  # evaluation without repeating detections (CK_SKIP_DETECTION=YES)
  with open(IMAGE_LIST_FILE, 'w') as f:
    f.write(json.dumps(processed_image_ids))

  test_time = time.time() - test_time_begin
  detect_avg_time = detect_time_total / images_processed
  load_avg_time = load_time_total / len(processed_image_ids)

  OPENME['setup_time_s'] = setup_time
  OPENME['test_time_s'] = test_time
  OPENME['graph_load_time_s'] = graph_load_time
  OPENME['images_load_time_s'] = load_time_total
  OPENME['images_load_time_avg_s'] = load_avg_time
  OPENME['detection_time_total_s'] = detect_time_total
  OPENME['detection_time_avg_s'] = detect_avg_time
  OPENME['avg_time_ms'] = detect_avg_time * 1000
  OPENME['avg_fps'] = 1.0 / detect_avg_time if detect_avg_time > 0 else 0

  return processed_image_ids


def evaluate(processed_image_ids, categories_list):
  # Convert annotations from original format of the dataset
  # to a format specific for a tool that will calculate metrics
  if DATASET_TYPE != METRIC_TYPE:
    print('\nConvert annotations from {} to {} ...'.format(DATASET_TYPE, METRIC_TYPE))
    annotations = converter_annotations.convert(ANNOTATIONS_PATH, 
                                                ANNOTATIONS_OUT_DIR,
                                                DATASET_TYPE,
                                                METRIC_TYPE)
  else: annotations = ANNOTATIONS_PATH

  # Convert detection results from our universal text format
  # to a format specific for a tool that will calculate metrics
  print('\nConvert results to {} ...'.format(METRIC_TYPE))
  results = converter_results.convert(DETECTIONS_OUT_DIR, 
                                      RESULTS_OUT_DIR,
                                      DATASET_TYPE,
                                      MODEL_DATASET_TYPE,
                                      METRIC_TYPE)

  # Run evaluation tool
  print('\nEvaluate metrics as {} ...'.format(METRIC_TYPE))
  if METRIC_TYPE == ck_utils.COCO:
    mAP, recall, all_metrics = calc_metrics_coco.evaluate_via_pycocotools(processed_image_ids, results, annotations)
  elif METRIC_TYPE == ck_utils.COCO_TF:
    mAP, recall, all_metrics = calc_metrics_coco.evaluate_via_tf(categories_list, results, annotations, FULL_REPORT)
  elif METRIC_TYPE == ck_utils.OID:
    mAP, _, all_metrics = calc_metrics_oid.evaluate(results,
        os.path.join(ANNOTATIONS_DIR, annotations), LABELMAP_FILE)
    recall = 'N/A'

  else:
    raise ValueError('Metrics type is not supported: {}'.format(METRIC_TYPE))

  OPENME['mAP'] = mAP
  OPENME['recall'] = recall
  OPENME['metrics'] = all_metrics


def print_header(s):
  print('\n' + '*'*80)
  print('* ' + s)
  print('*'*80)


def main(_):
  # Print settings
  print("Model frozen graph: " + FROZEN_GRAPH)
  print("Model label map file: " + LABELMAP_FILE)
  print("Model is for dataset: " + MODEL_DATASET_TYPE)

  print("Dataset images: " + IMAGES_DIR)
  print("Dataset annotations: " + ANNOTATIONS_PATH)
  print("Dataset type: " + DATASET_TYPE)

  print('Image count: {}'.format(IMAGE_COUNT))
  print("Metric type: " + METRIC_TYPE)
  print('Results directory: {}'.format(RESULTS_OUT_DIR))
  print("Temporary annotations directory: " + ANNOTATIONS_OUT_DIR)
  print("Detections directory: " + DETECTIONS_OUT_DIR)
  print("Result images directory: " + IMAGES_OUT_DIR)
  print('Save result images: {}'.format(SAVE_IMAGES))

  # Create category index
  category_index = label_map_util.create_category_index_from_labelmap(LABELMAP_FILE, use_display_name=True)
  categories_list = category_index.values() # array: [{"id": 88, "name": "teddy bear"}, ...]
  print('Categories: {}'.format(categories_list))

  # Run detection if needed
  print_header('Process images')
  if SKIP_DETECTION:
    print('\nSkip detection, evaluate previous results')
    with open(IMAGE_LIST_FILE, 'r') as f:
      processed_image_ids = json.load(f)
  else:
    processed_image_ids = detect(category_index)

  # Run evaluation
  print_header('Process results')
  evaluate(processed_image_ids, categories_list)

  # Store benchmark results
  with open('tmp-ck-timer.json', 'w') as o:
    json.dump(OPENME, o, indent=2, sort_keys=True)

  # Print metrics
  print('\nSummary:')
  print('-------------------------------')
  print('Graph loaded in {:.6f}s'.format(OPENME.get('graph_load_time_s', 0)))
  print('All images loaded in {:.6f}s'.format(OPENME.get('images_load_time_s', 0)))
  print('All images detected in {:.6f}s'.format(OPENME.get('detection_time_total_s', 0)))
  print('Average detection time: {:.6f}s'.format(OPENME.get('detection_time_avg_s', 0)))
  print('mAP: {}'.format(OPENME['mAP']))
  print('Recall: {}'.format(OPENME['recall']))
  print('--------------------------------\n')


if __name__ == '__main__':
  tf.app.run()
