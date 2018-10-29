from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annType='bbox'
annFile = '/home/ivan/CK-TOOLS/dataset-coco-2014/annotations/instances_val2014.json'

cocoGt=COCO(annFile)

resFile='/home/ivan/CK/ck-tensorflow/program/tensorflow-api/tmp/results/coco_by_coco_as_coco.json'

cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())

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
  return 0

def evaluate_kitti(results, annotations):
  return 0
