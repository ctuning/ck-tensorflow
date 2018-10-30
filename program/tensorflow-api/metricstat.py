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

  stat=[
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ',
    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ',
    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = '
  ]
  res=''
  for i in range(0,len(stat)):
    res += stat[i] + '{:.3f}'.format(cocoEval.stats[i]) + '\n'

  return res

def evaluate_kitti(results, annotations):
  return 0
