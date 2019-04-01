#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import json
import datetime

import ck_utils as helper

def convert(source_path, target_dir, type_from, type_to):
  '''
  Convert annotations from original format of the dataset
  to a format specific for a tool that will calculate metrics.

  Source annotations `source_path` can be a directory or a single file,
  depending on dataset type `type_from`.

  Returns whether target annotations directory `target_dir`
  or path to the new annotations file, depending on target format `type_to`.
  '''
  if type_from == helper.COCO and type_to == helper.COCO_TF:
    return os.getenv("CK_ENV_DATASET_LABELS_DIR")
  
  if type_from == helper.KITTI and type_to == helper.COCO:
    return convert_kitti_to_coco(source_path, target_dir)
    
  if type_from == helper.COCO and type_to == helper.KITTI:
    return convert_coco_to_kitti(source_path, target_dir)
    
  raise ValueError('Unknown how to convert between these annotation types ({} -> {})'.format(type_from, type_to))


def convert_kitti_to_coco(source_dir, target_dir):
  files = helper.get_files(source_dir)
  write_file = os.path.join(target_dir, 'kitti_to_coco_annotations.json')
  ann_counter = 0 # annotations counter
  now = datetime.datetime.now()
  body = {
    "info": {
        "year": now.year,
        "version": "0.0",
        "description": "Annotations converted from kitti- to coco-dataset",
        "contributor": "Unknown",
        "url": "",
        "date_created" : str(now)
      },
    "images": [],
    "annotations": [],
    "categories": [],
    "licenses" : [{"id": 0, "name": "None", "url": "" }],
  }
  images_array=[]
  annotations_array=[]
  for file_counter, file_name in enumerate(files):
    print('{}: {} of {}'.format(file_name, file_counter+1, len(files)))
    
    img_file_name = os.path.splitext(file_name)[0] + ".jpg"
    read_file = os.path.join(source_dir, file_name)

    with open(read_file, 'r') as rf:
      file_id = filename_to_id(file_name, 'kitti')
      width = 0
      height = 0
      file_item = {
        "id": file_id,
        "width" : width,
        "height" : height,
        "file_name" : img_file_name,
        "license" : 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": str(now),
      }

      images_array.append(file_item)

      for line in rf:
        str_array = line.split()
        class_name = str_array[0].lower()
        if class_name not in helper.KITTI_CLASSES:
          continue
        category_id = helper.KITTI_CLASSES[class_name]
        ann_counter += 1
        x1 = str_array[4]
        y1 = str_array[5]
        x2 = str_array[6]
        y2 = str_array[7]
        x = float(x1)
        y = float(y1)
        width = round(float(x2) - x, 2)
        height = round(float(y2) - y, 2)
        area = round(width * height, 2)
        annotation = {
                    "id" : ann_counter,
                    "image_id" : file_id,
                    "category_id" : category_id,
                    "segmentation" : [],
                    "area" : area,
                    "bbox" : [x, y, width, height],
                    "iscrowd" : 0,
                  }
        annotations_array.append(annotation)

  categories = []
  for idx in helper.KITTI2COCO:
    categories.append({
      "id" : helper.KITTI2COCO[idx][0],
      "name" : helper.KITTI2COCO[idx][1],
      "supercategory" : helper.KITTI2COCO[idx][2],
    })

  body["images"] = images_array
  body["annotations"] = annotations_array
  body["categories"] = categories

  with open(write_file, 'w') as wf:
    wf.write(json.dumps(body))
  return write_file


def convert_coco_to_kitti(dataset_annotations, target_dir):
  dataset = json.load(open(dataset_annotations, 'r'))

  images = dict()
  images_anns = dict()

  for img in dataset['images']:
    images[img['id']] = img['file_name']
    images_anns[img['id']] = list()

  for ann in dataset['annotations']:
    images_anns[ann['image_id']].append(ann)

  for image_id, image_file in images.items():
    label_file = os.path.join(target_dir, os.path.splitext(image_file)[0] + '.txt')
    with open(label_file, 'w') as lf:
      for ann in images_anns[image_id]:
        category_id = str(ann['category_id'])
        if category_id in helper.COCO2KITTI:
          name = re.sub(r'\s+', '', helper.COCO2KITTI[category_id][1]).capitalize()
        else:
          continue
        bbox = ann['bbox']
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        lf.write(name + ' 0.0 0 0.0 ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + ' 0.0  0.0  0.0  0.0  0.0  0.0  0.0\n')

  return target_dir
