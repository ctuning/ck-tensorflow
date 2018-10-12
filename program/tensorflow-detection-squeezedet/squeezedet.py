#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#  

import os
import cv2
import time
import json
import shutil
import subprocess

import numpy as np
import tensorflow as tf

# Model implementaion modules from package:demo-squeezedet-patched
from config import *
from nets import *

import evaluation
from evaluation import ImageResult, DetectionBox 


KITTI_EVAL_TOOL = os.getenv('CK_KITTI_EVAL_TOOL')
CUR_DIR = os.getcwd()
RESULTS_DIR = os.path.join(CUR_DIR, os.getenv('CK_RESULTS_DIR'))
# This must be called 'data' because of it is hardcoded in KITTI_EVAL_TOOL
RESULTS_DIR_DATA = os.path.join(RESULTS_DIR, 'data')
RESULTS_DIR_IMAGES = os.path.join(RESULTS_DIR, 'images')
IMAGE_DIR = os.getenv('CK_ENV_DATASET_IMAGE_DIR')
LABELS_DIR = os.getenv('CK_ENV_DATASET_KITTI_LABELS_DIR')
MODEL_NAME = os.getenv('CK_ENV_MODEL_SQUEEZEDET_ID')
WEIGHTS_FILE = os.getenv('CK_ENV_MODEL_SQUEEZEDET_MODEL')
BATCH_SIZE = int(os.getenv('CK_BATCH_SIZE'))
BATCH_COUNT = int(os.getenv('CK_BATCH_COUNT'))
IMAGE_COUNT = BATCH_SIZE * BATCH_COUNT
IMAGE_LIST_FILE = os.path.join(CUR_DIR, 'images.txt')
MEMORY_PERCENT = int(os.getenv('CK_TF_GPU_MEMORY_PERCENT'))
DETECTION_IOU = float(os.getenv('CK_DETECTION_IOU'))
FP_THRESHOLD = float(os.getenv('CK_FALSE_POSITIVE_THRESHOLD'))
GPU_ID = 0
CLEAN_RESULTS_DIR = True # debug option
PRINT_DETECTION_RESULTS = False # debug option
RESULTS_INFO = {} # to be populated during execution
CLASSES_METRICS = {} # to be populated during execution

MODEL = {}
MODEL_CONFIG = {}

DRAW_BOXES = True
BOXES_COLORS = {
  'car': (255, 191, 0),
  'cyclist': (0, 191, 255),
  'pedestrian': (255, 0, 191)
}


def get_class_color(class_name):
  if class_name in BOXES_COLORS:
    return BOXES_COLORS[class_name]
  return (0, 255, 0)


def make_model_and_config():
  known_models = {
    'squeezeDet': (SqueezeDet, kitti_squeezeDet_config),
    'squeezeDet+': (SqueezeDetPlus, kitti_squeezeDetPlus_config),
    'resnet50': (ResNet50ConvDet, kitti_res50_config),
    'vgg16': (VGG16ConvDet, kitti_vgg16_config)
  }
  if MODEL_NAME not in known_models:
    print('Selected neural net architecture is not supported: ' + MODEL_NAME)
    exit(1)
  model_class, config_func = known_models[MODEL_NAME]
  global MODEL
  global MODEL_CONFIG
  MODEL_CONFIG = config_func()
  MODEL_CONFIG.BATCH_SIZE = BATCH_SIZE
  # model parameters will be restored from checkpoint
  MODEL_CONFIG.LOAD_PRETRAINED_MODEL = False
  MODEL = model_class(MODEL_CONFIG, GPU_ID)
  print('Model configuration:')
  for key in ['IMAGE_HEIGHT', 'IMAGE_WIDTH', 'CLASSES', 'CLASS_NAMES', 
              'PROB_THRESH', 'PLOT_PROB_THRESH', 'NMS_THRESH',
              'TOP_N_DETECTION', 'BGR_MEANS']:
    print('{} = {}'.format(key, MODEL_CONFIG[key]))
  print('')


def box_transform(box):
  '''
  Converts a box of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax].
  '''
  cx, cy, w, h = box
  out_box = [[]]*4
  out_box[0] = cx-w/2
  out_box[1] = cy-h/2
  out_box[2] = cx+w/2
  out_box[3] = cy+h/2
  return out_box


def run_command(args_list):
  print(' '.join(args_list))
  process = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  output = process.communicate()[0]
  print(output)


def process_results(image_path, image_data, boxes, probes, classes):
  _, image_file = os.path.split(image_path)
  print('\nProcessing results for {}'.format(image_file))

  # Make detection results
  res = ImageResult()
  res.image_file = image_file
  res.label_file = os.path.splitext(image_file)[0] + '.txt'
  res.detections = []
  for box, prob, class_index in zip(boxes, probes, classes):
    d = DetectionBox()
    d.class_index = class_index
    d.class_name = MODEL_CONFIG.CLASS_NAMES[class_index].lower()
    d.prob = prob
    d.box = box_transform(box)
    res.detections.append(d)

  # Draw results boxes
  if DRAW_BOXES:
    result_image_file = os.path.join(RESULTS_DIR_IMAGES, image_file)
    print('Saving result image to {} ...'.format(result_image_file))
    for d in res.detections:
      if d.prob > MODEL_CONFIG.PLOT_PROB_THRESH:
        if PRINT_DETECTION_RESULTS:
          print(d.str())
        color = get_class_color(d.class_name)
        xmin, ymin, xmax, ymax = [int(x) for x in d.box]
        cv2.rectangle(image_data, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.putText(image_data, '{}: {:.2f}'.format(d.class_name, d.prob),
          (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    cv2.imwrite(result_image_file, image_data)

  # Save results data to process by KITTI evaluation tool
  if KITTI_EVAL_TOOL:
    result_data_file = os.path.join(RESULTS_DIR_DATA, res.label_file)
    print('Saving result data to {} ...'.format(result_data_file))
    with open(result_data_file, 'wt') as f:
      for d in res.detections:
        if PRINT_DETECTION_RESULTS:
          print(d.str())
        line = '{:s} -1 -1 0.0 {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {:.3f}' \
          .format(d.class_name, d.xmin(), d.ymin(), d.xmax(), d.ymax(), d.prob)
        f.write(line + '\n')

  return res


def load_image_list(d):
  assert os.path.isdir(d), 'Input dir does not exit'
  entries = [os.path.join(d, f) for f in os.listdir(d)]
  files = sorted([f for f in entries if os.path.isfile(f)])
  assert len(files) > 0, 'Input dir does not contain files'
  required_count = IMAGE_COUNT
  images = files[:required_count]
  if len(images) < required_count:
    for _ in range(required_count-len(images)):
      images.append(images[-1])
  with open(IMAGE_LIST_FILE, 'wt') as f:
    for path in images:
      f.write('{}\n'.format(os.path.splitext(os.path.split(path)[1])[0]))
  return images


def run_kitti_evaluation():
  if not KITTI_EVAL_TOOL: return
  
  print('\n---------------')
  print('Run KITTI evaluation tool ...')
  # CK_RESULTS_DIR should be a dir containig `data` subdir,
  # because of hardcode in kitti evaluation tool
  os.putenv('CK_RESULTS_DIR', RESULTS_DIR)
  os.putenv('CK_KITTI_LABELS_DIR', LABELS_DIR)
  os.putenv('CK_IMAGE_LIST_FILE', IMAGE_LIST_FILE)
  os.putenv('CK_IMAGE_COUNT', str(IMAGE_COUNT))
  os.chdir(RESULTS_DIR);
  run_command([KITTI_EVAL_TOOL])
  os.chdir(CUR_DIR)

  def load_AP(class_name):
    file_name = 'stats_{}_ap.txt'.format(class_name.lower())
    print('Loading {} ...'.format(file_name))
    file_name = os.path.join(RESULTS_DIR, file_name)
    if not os.path.exists(file_name):
      print('WARNING: File does not exist')
      return 0, 0, 0
    with open(file_name, 'r') as f:
      lines = f.readlines()
    if len(lines) != 3:
      print('ERROR: Line number should be 3')
      return 0, 0, 0
    easy = float(lines[0].split('=')[1].strip())
    moderate = float(lines[1].split('=')[1].strip())
    hard = float(lines[2].split('=')[1].strip())
    return easy, moderate, hard
  
  print('\nProcessing evaluation results ...')
  total = 0
  for class_name in MODEL_CONFIG.CLASS_NAMES:
    print('\n{}'.format(class_name.upper()))
    easy, moderate, hard = load_AP(class_name)
    total += easy + moderate + hard
    print('easy: AP={:.3f}'.format(easy))
    print('moderate: AP={:.3f}'.format(moderate))
    print('hard: AP={:.3f}'.format(hard))
    CLASSES_METRICS[class_name]['AP_kitti'] = {
      'easy': easy, 'moderate': moderate, 'hard': hard
    }
  mAP = total / float(len(MODEL_CONFIG.CLASS_NAMES)) / 3.0
  RESULTS_INFO['mAP_kitti'] = mAP


def run_default_evaluation(results):
  print('\n---------------')
  print('Run default evaluation ...')
  evaluation.LABELS_DIR = LABELS_DIR
  evaluation.CLASS_NAMES = MODEL_CONFIG.CLASS_NAMES
  evaluation.DETECTION_IOU = DETECTION_IOU
  evaluation.FP_THRESHOLD = FP_THRESHOLD
  evaluation.evaluate_images(results, RESULTS_INFO, CLASSES_METRICS)


def main(_):
  print('Images dir: {}'.format(IMAGE_DIR))
  print('Results dir: {}'.format(RESULTS_DIR))
  print('Model name: {}'.format(MODEL_NAME))
  print('Model weights: {}'.format(WEIGHTS_FILE))
  print('Batch size: {}'.format(BATCH_SIZE))
  print('Batch count: {}'.format(BATCH_COUNT))
  print('Memory limit: {}%'.format(MEMORY_PERCENT))
  print('KITTI evaluation tool: {}'.format(KITTI_EVAL_TOOL))
  if not KITTI_EVAL_TOOL:
    print('    (Not found, only default evaluation is available)')
  print('\n----------------------------')

  # Prepare detection results dir
  if CLEAN_RESULTS_DIR:
    if os.path.isdir(RESULTS_DIR):
      shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)
    os.mkdir(RESULTS_DIR_DATA)
    if DRAW_BOXES:
      os.mkdir(RESULTS_DIR_IMAGES)

  # Load processing image filenames
  image_list = load_image_list(IMAGE_DIR)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_PERCENT/100.0)
  config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

  with tf.Graph().as_default(), tf.Session(config=config) as sess:
    # Make model config and implementation
    begin_time = time.time()
    make_model_and_config()
    print('Model created in {}s'.format((time.time() - begin_time)))

    # Restore model parameters
    begin_time = time.time()
    saver = tf.train.Saver(MODEL.model_params)
    saver.restore(sess, WEIGHTS_FILE)
    print('Params restored in {}s'.format((time.time() - begin_time)))

    # Detect all batches
    total_time = 0
    image_index = 0
    times_measured = 0
    boxes_detected = 0
    images_results = []
    for batch_index in range(BATCH_COUNT):
      # Load and preprocess batch images
      image_data = []
      batch_data = []
      batch_scales = []
      for _ in range(BATCH_SIZE):
        img = cv2.imread(image_list[image_index])
        image_data.append(img)
        orig_h, orig_w, _ = [float(v) for v in img.shape]
        w_scale = MODEL_CONFIG.IMAGE_WIDTH / orig_w
        h_scale = MODEL_CONFIG.IMAGE_HEIGHT / orig_h
        batch_scales.append((w_scale, h_scale))
        img = img.astype(np.float32, copy=False)
        img = cv2.resize(img, (MODEL_CONFIG.IMAGE_WIDTH, MODEL_CONFIG.IMAGE_HEIGHT))
        batch_data.append(img - MODEL_CONFIG.BGR_MEANS)
        image_index += 1

      # Detect batch
      begin_time = time.time()
      feed = {MODEL.image_input: batch_data}
      batch_boxes, batch_probs, batch_classes = sess.run(
        [MODEL.det_boxes, MODEL.det_probs, MODEL.det_class], feed_dict=feed)
      detect_time = time.time() - begin_time
      print('Batch detected in {}s'.format(detect_time))

      # Exclude first batch from averaging
      if batch_index > 0 or BATCH_COUNT == 1:
        total_time += detect_time
        times_measured += BATCH_SIZE

      # Process batch results
      assert len(batch_boxes) == len(batch_probs) \
         and len(batch_probs) == len(batch_classes) \
         and len(batch_classes) == BATCH_SIZE
      for i in range(BATCH_SIZE):
        # Rescale boxes to original image size
        batch_boxes[i, :, 0::2] /= batch_scales[i][0]
        batch_boxes[i, :, 1::2] /= batch_scales[i][1]

        # Internal model filtering
        boxes, probes, classes = MODEL.filter_prediction(batch_boxes[i], batch_probs[i], batch_classes[i])
        boxes_detected += len(boxes)

        # Write batch results
        img_index = batch_index * BATCH_SIZE + i
        res = process_results(image_list[img_index], image_data[i], boxes, probes, classes)
        images_results.append(res)

  print('\n----------------------------')
  print('Validation ...')
  for class_name in MODEL_CONFIG.CLASS_NAMES:
    CLASSES_METRICS[class_name] = {}
  run_kitti_evaluation()
  run_default_evaluation(images_results)

  print('\n----------------------------')
  avg_time = total_time/times_measured
  avg_fps = 1 / avg_time
  avg_detects = float(boxes_detected)/len(images_results)
  print('Images processed: {}'.format(len(images_results)))
  print('Boxes detected: {}'.format(boxes_detected))
  print('Average time: {:.3f}s'.format(avg_time));
  print('Average FPS: {:.3f}'.format(avg_fps))
  print('Average detections per image: {:.3f}'.format(avg_detects))
  print('Mean average precision: {:.3f}'.format(RESULTS_INFO['mAP']))
  if 'mAP_kitti' in RESULTS_INFO:
    print('Mean average precision (by KITTI): {:.3f}'.format(RESULTS_INFO['mAP_kitti']))

  RESULTS_INFO['avg_time_s'] = avg_time
  RESULTS_INFO['avg_fps'] = avg_fps
  RESULTS_INFO['avg_detects'] = avg_detects
  RESULTS_INFO['classes'] = CLASSES_METRICS
  with open('tmp-ck-timer.json', 'w') as f:
    json.dump(RESULTS_INFO, f, indent=2)

if __name__ == '__main__':
  tf.app.run()
