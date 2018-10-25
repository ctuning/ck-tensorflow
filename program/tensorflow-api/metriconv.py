import os
import json
import re
import datetime

KITTI2COCO = {
  "1":[1, "person", "person"],
  "2":[3, "car", "vehicle"]
}

KITTI_CLASSES = {
  "pedestrian": 1,
  "car":2,
}

COCO2KITTI = {
  "1":[1, "pedestrian"],
  "3":[2, "car"]
}

def convert_annotations(dataset_annotations, target_annotations, type_from, type_to):

  if type_from == type_to:
    return dataset_annotations
  elif type_from == 'kitti' and type_to=='coco':
    return convert_annotations_kitti_to_coco(dataset_annotations, target_annotations)
  elif type_from == 'coco' and type_to=='kitty':
    return convert_annotations_coco_to_kitti(dataset_annotations, target_annotations)
  else:
    return ''

def convert_results(pre_results, target_dir, dataset_type, model_dataset_type, target_metric_type):
  if target_metric_type == 'none':
    return pre_results
  elif target_metric_type == 'coco' and \
      (model_dataset_type == 'kitti' or model_dataset_type == 'coco'):
    return convert_results_kitti_and_coco_to_coco(pre_results, target_dir, dataset_type, model_dataset_type)
  elif target_metric_type == 'kitti' and \
      (model_dataset_type == 'kitti' or model_dataset_type == 'coco'):
    return convert_results_coco_and_kitti_to_kitti(pre_results, target_dir, dataset_type, model_dataset_type)
  return ''

def convert_results_coco_and_kitti_to_kitti(pre_results, target_dir, dataset_type, model_dataset_type):
  FILES=[f for f in os.listdir(pre_results) if os.path.isfile(os.path.join(pre_results, f))]
  file_counter = 0
  print('Progress: ')
  for file_name in FILES:
    file_counter += 1
    #file_name = os.path.splitext(any_file)[0]
    read_file = os.path.join(pre_results, file_name)
    write_file = os.path.join(target_dir, file_name)
    with open(read_file, 'r') as rf:
      rf.readline() #image size
      with open(write_file, 'w') as wf:
        for line in rf:
          if model_dataset_type == 'coco':
            res = result_coco_to_kitti(line)
          else:
            res = result_kitti_to_kitti(line)
          if (res):
            wf.write(res)  
    print(file_name + ": " + `file_counter` +" of " + `len(FILES)`)

  return target_dir

def pre_splitter(str):
  splitted = str.split()
  y1 = splitted[0]
  x1 = splitted[1]
  y2 = splitted[2]
  x2 = splitted[3]
  score = splitted[4]
  class_id = splitted[5]
  class_name = ' '.join(splitted[6:])
  return (x1, y1, x2, y2, score, class_id, class_name)

def result_coco_to_kitti(str):
  (x1, y1, x2, y2, score, class_id, class_name) = pre_splitter(str)
  if class_id in COCO2KITTI:
    res = '{} -1 -1 0.0 {} {} {} {} 0.0 0.0 0.0 0.0'\
      ' 0.0 0.0 0.0 {}\n'.format(COCO2KITTI[class_id][1], x1, y1, \
      x2, y2, score)
  else:
    res=''
  return res

def result_kitti_to_kitti(str):
  (x1, y1, x2, y2, score, class_id, class_name) = pre_splitter(str)
  res = '{} -1 -1 0.0 {} {} {} {} 0.0 0.0 0.0 0.0'\
    ' 0.0 0.0 0.0 {}\n'.format(class_name, x1, y1, x2, y2, score)
  return res

def convert_results_kitti_and_coco_to_coco(pre_results, target_dir, dataset_type, model_dataset_type):
  FILES=[f for f in os.listdir(pre_results) if os.path.isfile(os.path.join(pre_results, f))]
  file_counter = 0
  write_file = os.path.join(target_dir, dataset_type + '_by_' + model_dataset_type + '_as_coco.json')
  res_array = []
  print('Progress results: ')
  for file_name in FILES:
    file_counter += 1
    #file_name = os.path.splitext(any_file)[0]
    read_file = os.path.join(pre_results, file_name)
    file_id = filename_to_id(file_name, dataset_type)

    with open(read_file, 'r') as rf:
      rf.readline() #image size
      
      for line in rf:
        (x1, y1, x2, y2, score, class_id, class_name) = pre_splitter(line)
        
        if model_dataset_type == 'coco':
          category_id = int(class_id)
        else:
          category_id = KITTI2COCO[class_id][0]
        x = float(x1)
        y = float(y1)
        x_width = round(float(x2) - x, 2)
        y_height = round(float(y2) - y, 2)
        new_res = {
                    "image_id" : file_id,
                    "category_id" : category_id,
                    "bbox" : [x, y, x_width, y_height],
                    "score" : float(score),
                  }
        res_array.append(new_res)
 
    print(file_name + ": " + `file_counter` +" of " + `len(FILES)`)

  with open(write_file, 'w') as wf:
    wf.write(json.dumps(res_array))
  return write_file

def filename_to_id(file_name, from_type):
  short_name = os.path.splitext(file_name)[0]
  if from_type == 'kitti':
    id = int(short_name)
  elif from_type == 'coco':
    id = int(re.split(r'_', short_name)[2])
  else:
    id = int(os.path.splitext(file_name)[0])
  return id

def convert_annotations_kitti_to_coco(dataset_annotations, target_annotations):
  FILES=[f for f in os.listdir(dataset_annotations) if os.path.isfile(os.path.join(dataset_annotations, f))]
  write_file = os.path.join(target_annotations, 'kitti_to_coco_annotations.json')
  print('Progress annotations: ')
  file_counter = 0
  ann_counter = 0 # annotations counter
  now = datetime.datetime.now()
  body={
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
  for file_name in FILES:
    file_counter += 1
    img_file_name = os.path.splitext(file_name)[0] + ".jpg"
    read_file = os.path.join(dataset_annotations, file_name)

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
        #(x1, y1, x2, y2, score, class_id, class_name) = pre_splitter(line)
        str_array = line.split()
        class_name = str_array[0].lower()
        if class_name not in KITTI_CLASSES:
          continue
        category_id = KITTI_CLASSES[class_name]
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

    print(file_name + ": " + `file_counter` +" of " + `len(FILES)`)
  
  categories=[]
  for idx in KITTI2COCO:
    categories.append({
      "id" : KITTI2COCO[idx][0],
      "name" : KITTI2COCO[idx][1],
      "supercategory" : KITTI2COCO[idx][2],
    })

  body["images"] = images_array
  body["annotations"] = annotations_array
  body["categories"] = categories

  with open(write_file, 'w') as wf:
    wf.write(json.dumps(body))
  return write_file

def convert_annotations_coco_to_kitti(dataset_annotations, target_annotations):
  return ''