#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import json
import sys

ENV_JSON = 'env.json'

with open(ENV_JSON, 'r') as f:
  ENV = json.load(f)
for path in ENV['PYTHONPATH'].split(':'):
  sys.path.insert(0, path)
sys.path.append(os.path.dirname(__file__))

# Silence a warning (https://github.com/tensorflow/models/issues/3786)
import matplotlib; matplotlib.use('Agg')
import ck_utils
import converter_results
import converter_annotations
import calc_metrics_kitti
import calc_metrics_coco
import calc_metrics_oid
from object_detection.utils import label_map_util

LABELMAP_FILE = ENV['LABELMAP_FILE']
MODEL_DATASET_TYPE = ENV['MODEL_DATASET_TYPE']
DATASET_TYPE = ENV['DATASET_TYPE']
ANNOTATIONS_PATH = ENV['ANNOTATIONS_PATH']
METRIC_TYPE = ENV['METRIC_TYPE']
IMAGES_OUT_DIR = ENV['IMAGES_OUT_DIR']
DETECTIONS_OUT_DIR = ENV['DETECTIONS_OUT_DIR']
ANNOTATIONS_OUT_DIR = ENV['ANNOTATIONS_OUT_DIR']
RESULTS_OUT_DIR = ENV['RESULTS_OUT_DIR']
FULL_REPORT = ENV['FULL_REPORT']
IMAGE_LIST_FILE = ENV['IMAGE_LIST_FILE']
TIMER_JSON = ENV['TIMER_JSON']

def ck_postprocess(i):
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

    print(processed_image_ids, results, annotations)
    if METRIC_TYPE == ck_utils.COCO:
      mAP, recall, all_metrics = calc_metrics_coco.evaluate_via_pycocotools(processed_image_ids, results, annotations)
    elif METRIC_TYPE == ck_utils.COCO_TF:
      mAP, recall, all_metrics = calc_metrics_coco.evaluate_via_tf(categories_list, results, annotations, FULL_REPORT)
    elif METRIC_TYPE == ck_utils.OID:
      mAP, _, all_metrics = calc_metrics_oid.evaluate(results, annotations, LABELMAP_FILE, FULL_REPORT)
      recall = 'N/A'
    elif METRIC_TYPE == ck_utils.KITTI:
      mAP, _, all_metrics = calc_metrics_kitti.evaluate(DETECTIONS_OUT_DIR, annotations, LABELMAP_FILE, processed_image_ids)
      recall = 'N/A'

    else:
      raise ValueError('Metrics type is not supported: {}'.format(METRIC_TYPE))

    OPENME['mAP'] = mAP
    OPENME['recall'] = recall
    OPENME['metrics'] = all_metrics

    return

  OPENME = {}

  with open(IMAGE_LIST_FILE, 'r') as f:
    processed_image_ids = json.load(f)
  
  if os.path.isfile(TIMER_JSON):
    with open(TIMER_JSON, 'r') as f:
      OPENME = json.load(f)
  
  # Run evaluation
  ck_utils.print_header('Process results')
  category_index = label_map_util.create_category_index_from_labelmap(LABELMAP_FILE, use_display_name=True)
  categories_list = category_index.values()
  evaluate(processed_image_ids, categories_list)

  # Store benchmark results
  with open(TIMER_JSON, 'w') as o:
    json.dump(OPENME, o, indent=2, sort_keys=True)

  # Print metrics
  print('\nSummary:')
  print('-------------------------------')
  print('Graph loaded in {:.6f}s'.format(OPENME.get('graph_load_time_s', 0)))
  print('All images loaded in {:.6f}s'.format(OPENME.get('images_load_time_total_s', 0)))
  print('All images detected in {:.6f}s'.format(OPENME.get('detection_time_total_s', 0)))
  print('Average detection time: {:.6f}s'.format(OPENME.get('detection_time_avg_s', 0)))
  print('mAP: {}'.format(OPENME['mAP']))
  print('Recall: {}'.format(OPENME['recall']))
  print('--------------------------------\n')

  return {'return': 0}
