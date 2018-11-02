#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import tensorflow as tf

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from object_detection import eval_util
from object_detection.core import standard_fields as fields
from object_detection.protos import eval_pb2

def evaluate_single(categories_list, image_id, image_data, output_dict, sess):
  input_data_fields = fields.InputDataFields
  detection_fields = fields.DetectionResultFields
    
  eval_config = eval_pb2.EvalConfig()
  eval_config.metrics_set.extend(['coco_detection_metrics'])

  detection_masks = tf.ones(shape=[1, 1, 20, 20], dtype=tf.float32)

  detections = {
    detection_fields.detection_boxes: output_dict['detection_classes'],
    detection_fields.detection_scores: output_dict['detection_scores'],
    detection_fields.detection_classes: output_dict['detection_classes'],
    detection_fields.num_detections: output_dict['num_detections'],
    detection_fields.detection_masks: detection_masks,
  }

  # TODO: load from annotations
  groundtruth_boxes = tf.constant([[0., 0., 1., 1.]])
  groundtruth_classes = tf.constant([1])
  groundtruth_instance_masks = tf.ones(shape=[1, 20, 20], dtype=tf.uint8)

  groundtruth = {
    input_data_fields.groundtruth_boxes: groundtruth_boxes,
    input_data_fields.groundtruth_classes: groundtruth_classes,
    input_data_fields.groundtruth_instance_masks: groundtruth_instance_masks
  }

  eval_dict = eval_util.result_dict_for_single_example(image_data,
                                                       image_id,
                                                       detections,
                                                       groundtruth)

  metric_ops = eval_util.get_eval_metric_ops_for_evaluators(eval_config,
                                                            categories_list,
                                                            eval_dict)
  _, update_op = metric_ops['DetectionBoxes_Precision/mAP']

  metrics = {}
  for key, (value_op, _) in metric_ops.iteritems():
    metrics[key] = value_op
  sess.run(update_op)
  metrics = sess.run(metrics)
  print(metrics)


def evaluate(image_ids_list, results_dir, annotations_file):
  annType = 'bbox'
  cocoGt = COCO(annotations_file)
  cocoDt = cocoGt.loadRes(results_dir)
  cocoEval = COCOeval(cocoGt, cocoDt, annType)
  cocoEval.params.imgIds = image_ids_list
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()

  stat = {
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ': cocoEval.stats[0],
    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ': cocoEval.stats[1],
    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ': cocoEval.stats[2],
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ': cocoEval.stats[3],
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ': cocoEval.stats[4],
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ': cocoEval.stats[5],
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ': cocoEval.stats[6],
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ': cocoEval.stats[7],
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ': cocoEval.stats[8],
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ': cocoEval.stats[9],
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ': cocoEval.stats[10],
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ': cocoEval.stats[11]
  }

  return stat
