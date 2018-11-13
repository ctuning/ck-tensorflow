#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os

import numpy as np

from object_detection.core import standard_fields
from object_detection.legacy import evaluator
from object_detection.utils import label_map_util

import ck_utils

def new_detection(key):
  return {
    'key': key,
    'groundtruth_boxes': np.empty([0,4]),
    'groundtruth_classes': np.array([],dtype=int),
    'groundtruth_difficult': None,
    'groundtruth_group_of': np.array([],dtype=int),
    'groundtruth_image_classes': None,
    'detection_boxes': np.empty([0,4]),
    'detection_classes': np.array([],dtype=int),
    'detection_scores': np.array([],dtype=float),
  }

def fill_detection(detection, file_name):
  with open(file_name, 'r') as f:
    size = f.readline().split()
    x = float(size[0])
    y = float(size[1])
    cnt = 0
    for line in f:
      det = ck_utils.Detection(line)
      cnt += 1
      detection['detection_boxes'] = np.append(detection['detection_boxes'], [[det.x1/x, det.y1/y, det.x2/x, det.y2/y]], 0)
      detection['detection_classes'] = np.append(detection['detection_classes'], det.class_id)
      detection['detection_scores'] = np.append(detection['detection_scores'], det.score)
    # Detections dict can't be empty even there is not detection for this image
    if cnt == 0:
      detection['detection_boxes'] = np.array([[0.0, 0.0, 0.0, 0.0]])
      detection['detection_classes'] = np.array([0])
      detection['detection_scores'] = np.array([0.0])
  return detection

def get_annotations(annotations, class_name_to_id_map):
  '''
  Get annotations from file to dictionary in memory

  Args:
    annotations: groundtruth (file with annotations)
    class_name_to_id_map: dictionary for classes, text id => number id 

  Returns:
    Dictionary with annotations
  '''
  res = {}

  with open(annotations, 'r') as ann:
    ann.readline() #header
    for line in ann:
      str = line.split(',')
      key = str[0]
      img_class = class_name_to_id_map[str[2]]
      x_min = float(str[4])
      x_max = float(str[5])
      y_min = float(str[6])
      y_max = float(str[7])
      group = int(str[10])
      if key not in res:
        res[key] = []
      res[key].append({
        'class': img_class,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'group': group
      })

  return res

def fill_annotations(detection, annotations):
  for ann in annotations:
    detection['groundtruth_group_of'] = np.append(detection['groundtruth_group_of'], ann['group'])
    detection['groundtruth_boxes'] = np.append(detection['groundtruth_boxes'],
      [[ann['x_min'], ann['y_min'], ann['x_max'], ann['y_max']]], 0)
    detection['groundtruth_classes'] = np.append(detection['groundtruth_classes'], ann['class'])
  return detection


def evaluate(res_dir, annotations, label_map_path, full_report):
  '''
  Calculate OID metrics via evaluator class included in TF models repository
  https://github.com/tensorflow/models/tree/master/research/object_detection/metrics

  Reads pre-computed object detections and groundtruth.

  Args:
    res_dir: pre-computed object detections directory
    annotations: groundtruth (file with annotations)
    label_map_path: labelmap file

  Returns:
    Evaluated detections metrics.
  '''
  class EvaluatorConfig:
    metrics_set = ['open_images_V2_detection_metrics']

  eval_config = EvaluatorConfig()

  categories = label_map_util.create_categories_from_labelmap(label_map_path)
  class_map = label_map_util.get_label_map_dict(label_map_path, False, False)

  object_detection_evaluators = evaluator.get_evaluators(
      eval_config, categories)
  # Support a single evaluator
  object_detection_evaluator = object_detection_evaluators[0]

  print('Loading annotations...')
  ann = get_annotations(annotations, class_map)

  files = ck_utils.get_files(res_dir)
  for file_index, file_name in enumerate(files):
    if full_report:
      print('Loading detections and annotations for {} ({} of {}) ...'.format(file_name, file_index+1, len(files)))
    elif (file_index+1) % 100 == 0:
      print('Loading detections and annotations: {} of {} ...'.format(file_index+1, len(files)))
    det_file = os.path.join(res_dir, file_name)
    key = os.path.splitext(file_name)[0]
    detection = new_detection(key)
    fill_annotations(detection, ann[key])
    fill_detection(detection, det_file)

    object_detection_evaluator.add_single_ground_truth_image_info(
        detection[standard_fields.DetectionResultFields.key],
        detection)
    object_detection_evaluator.add_single_detected_image_info(
        detection[standard_fields.DetectionResultFields.key],
        detection)

  all_metrics = object_detection_evaluator.evaluate()
  mAP = all_metrics['OpenImagesV2_Precision/mAP@0.5IOU']

  return mAP, 0, all_metrics
