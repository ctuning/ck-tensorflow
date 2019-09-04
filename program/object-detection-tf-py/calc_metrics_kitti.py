#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
from object_detection.utils import label_map_util
from object_detection.utils.object_detection_evaluation import PascalDetectionEvaluator as evaluator
from object_detection.core import standard_fields
import numpy as np
import ck_utils

gt_field = standard_fields.InputDataFields
det_field = standard_fields.DetectionResultFields


def load_groundtruth(file_path, class_name_to_id_map):
  boxes = []
  classes = []
  with open(file_path, 'r') as f:
    for line in f:
      gt = ck_utils.Groundtruth(line)
      if gt.class_name.lower() in class_name_to_id_map.keys():
        boxes.append([gt.x1, gt.y1, gt.x2, gt.y2])
        classes.append(class_name_to_id_map[gt.class_name.lower()])
  if boxes:
    return {
      gt_field.groundtruth_boxes: np.array(boxes),
      gt_field.groundtruth_classes: np.array(classes)
    }


def load_detections(file_path):
  boxes = []
  scores = []
  classes = []
  with open(file_path, 'r') as f:
    f.readline() # first line is image size
    for line in f:
      det = ck_utils.Detection(line)
      boxes.append([det.x1, det.y1, det.x2, det.y2])
      scores.append(det.score)
      classes.append(det.class_id)
  # Detections dict can't be empty even there is not detection for this image
  if not boxes:
    boxes = [[0,0,0,0]]
    scores = [0]
    classes = [0]
  return {
    det_field.detection_boxes: np.array(boxes),
    det_field.detection_scores: np.array(scores),
    det_field.detection_classes: np.array(classes)
  }




def evaluate(results_dir, annotations_dir,labelmap,processed_img_ids):
  category_index = label_map_util.create_category_index_from_labelmap(labelmap, use_display_name=True)
  cat_list = []
  cat_names = {}
  for cat in category_index:
    cat_list.append(category_index[cat])
    cat_names[category_index[cat]['name'].lower()] = category_index[cat]['id']

  total_dets_count = 0
  total_gts_count = 0
  not_found_gts = []


  valu = evaluator(cat_list)
### use annotation dir to check if is drive or not
  is_drive = False
  if 'drive' in annotations_dir:
    print ("DRIVE DATASET")
    is_drive = True

 #### build groundtruth dict
  for img in processed_img_ids:
    if is_drive:
      filepath = os.path.join(annotations_dir ,"{:010d}.txt".format(int(img)))
    else:
      filepath = os.path.join(annotations_dir ,"{:06d}.txt".format(int(img)))
    gts = load_groundtruth(filepath, cat_names)
    if not gts:
      not_found_gts.append(img)
      continue 
    #print (gts[gt_field.groundtruth_boxes], gts[gt_field.groundtruth_classes])          
    if is_drive:
      det_file = os.path.join(results_dir ,"{:010d}.txt".format(int(img)))
    else:
      det_file = os.path.join(results_dir ,"{:06d}.txt".format(int(img)))
    #print(det_file)
    dets = load_detections(det_file)

    gts_count = gts[gt_field.groundtruth_boxes].shape[0]
    dets_count = dets[det_field.detection_boxes].shape[0]
    total_gts_count += gts_count
    total_dets_count += dets_count

    #if full_report:
    #print('  Detections: {}'.format(dets_count))
    #print('  Groundtruth: {}'.format(gts_count))

    # Groundtruth should be added first, as adding image checks if there is groundtrush for it
    valu.add_single_ground_truth_image_info(image_id=img, groundtruth_dict=gts)
    valu.add_single_detected_image_info(image_id=img, detections_dict=dets)
        

  all_metrics = valu.evaluate()

  if not_found_gts:
    print('Groundtrush not found for {} results:'.format(len(not_found_gts)))
    for file_name in not_found_gts:
      print('    {}'.format(file_name))


 
  mAP = all_metrics['PascalBoxes_Precision/mAP@0.5IOU']
  recall = 'N/A' 
  return mAP, recall, all_metrics

