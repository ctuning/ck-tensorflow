#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
