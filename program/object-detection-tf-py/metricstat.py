from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(image_list, results, annotations, target_metric_type):
  if target_metric_type == 'coco':
    return evaluate_coco(image_list, results, annotations)
  elif target_metric_type == 'kitti':
    return evaluate_kitti(results, annotations)
  else:
    raise ValueError('Metric type "' + target_metric_type +'" not realized yet')

def evaluate_coco(image_list, results, annotations):
  annType='bbox'
  cocoGt=COCO(annotations)
  cocoDt=cocoGt.loadRes(results)
  cocoEval = COCOeval(cocoGt,cocoDt, annType)
  cocoEval.params.imgIds  = image_list
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()

  stat={
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

def evaluate_kitti(results, annotations):
  return 0
