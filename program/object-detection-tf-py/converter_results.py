#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import json

import ck_utils
import converter_utils as helper

def convert(detections_dir, target_dir, dataset_type, model_dataset_type, metric_type):
  '''
  Convert detection results from our universal text format
  to a format specific for a tool that will calculate metrics.

  Returns whether results directory or path to the new results file,
  depending on target results format.
  '''
  detection_files = ck_utils.get_files(detections_dir)

  if metric_type == helper.COCO:
    return convert_to_coco(detection_files, detections_dir, target_dir, dataset_type, model_dataset_type)
    
  if metric_type == helper.KITTI:
    return convert_to_kitti(detection_files, detections_dir, target_dir, model_dataset_type)
    
  raise Exception('Unknown target results format: {}'.format(metric_type))


def convert_to_kitti(detection_files, detections_dir, target_dir, model_dataset_type):
  for file_name in detection_files:
    read_file = os.path.join(detections_dir, file_name)
    write_file = os.path.join(target_dir, file_name)
    with open(read_file, 'r') as rf, open(write_file, 'w') as wf:
      rf.readline() # first line is image size
      for line in rf:
        det = Detection(line)
        res = detection_to_kitti_string(model_dataset_type)
        if (res):
          wf.write(res)  
  return target_dir


def convert_to_coco(detection_files, detections_dir, target_dir, dataset_type, model_dataset_type):
  res_array = []
  for file_name in detection_files:
    read_file = os.path.join(detections_dir, file_name)
    file_id = helper.filename_to_id(file_name, dataset_type)
    with open(read_file, 'r') as rf:
      rf.readline() # first line is image size
      for line in rf:
        det = Detection(line)
        res = detection_to_coco_object(det, model_dataset_type, file_id)
        if (res):
          res_array.append(res)
  results_file = os.path.join(target_dir, 'coco_results.json')
  with open(results_file, 'w') as f:
    f.write(json.dumps(res_array, indent=2, sort_keys=False))
  return results_file


class Detection:
  def __init__(self, line):
    splitted = line.split()
    self.y1 = splitted[0]
    self.x1 = splitted[1]
    self.y2 = splitted[2]
    self.x2 = splitted[3]
    self.score = splitted[4]
    self.class_id = splitted[5]
    self.class_name = ' '.join(splitted[6:])


def detection_to_kitti_string(det, model_dataset_type):
  '''
  Returns result line in the format expected by kitti-eval-tool
  '''
  class_name = ''

  if model_dataset_type == helper.KITTI:
    class_name = det.class_name

  elif model_dataset_type == helper.COCO:
    if det.class_id in helper.COCO2KITTI:
      class_name = helper.COCO2KITTI[class_id][1]

  if not class_name: return ''
      
  return '{} -1 -1 0.0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {}\n'\
    .format(class_name, det.x1, det.y1, det.x2, det.y2, det.score)


def detection_to_coco_object(det, model_dataset_type, file_id):
  '''
  Returns result object in COCO format
  '''
  category_id = None
  
  if model_dataset_type == helper.COCO:
    category_id = int(det.class_id)

  elif model_dataset_type == helper.KITTI:
    category_id = helper.KITTI2COCO[det.class_id][0]

  if not category_id: return None
    
  x = float(det.x1)
  y = float(det.y1)
  w = round(float(det.x2) - x, 2)
  h = round(float(det.y2) - y, 2)
  return {
    "image_id" : file_id,
    "category_id" : category_id,
    "bbox" : [x, y, w, h],
    "score" : float(det.score),
  }
