#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import re
import shutil

KITTI = 'kitti'
COCO = 'coco'
OID = 'openimages'

# This is only used for METRIC_TYPE
COCO_TF = 'coco_tf'

# KITTI classes are taken from
# https://github.com/tensorflow/models/blob/master/research/object_detection/data/kitti_label_map.pbtxt
# and only contain two classes - 'pedestrian' and 'car'. But the official KITTI evaluation tool
# (implemented as `ck-tensorflow:program:kitti-eval-tool`) knows about `cyclist` class too.
KITTI_CLASSES = {
  "pedestrian": 1,
  "car": 2,
}

KITTI2COCO = {
  "1": [1, "person", "person"],
  "2": [3, "car", "vehicle"]
}

COCO2KITTI = {
  "1": [1, "pedestrian"],
  "3": [2, "car"]
}

def prepare_dir(dir_path):
  '''
  Recreate a directory
  '''
  if os.path.isdir(dir_path):
    shutil.rmtree(dir_path)
  os.mkdir(dir_path)


def get_files(dir_path):
  return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


def load_image_list(images_dir, images_count, skip_images):
  '''
  Load list of images to be processed
  '''
  assert os.path.isdir(images_dir), 'Input dir does not exit'
  files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
  files = [f for f in files if re.search(r'\.jpg$', f, re.IGNORECASE)
                            or re.search(r'\.jpeg$', f, re.IGNORECASE)
                            or re.search(r'\.png$', f, re.IGNORECASE)]
  assert len(files) > 0, 'Input dir does not contain image files'
  files = sorted(files)[skip_images:]
  assert len(files) > 0, 'Input dir does not contain more files'
  images = files[:images_count]
  # Repeat last image to make full last batch
  if len(images) < images_count:
    for _ in range(images_count-len(images)):
      images.append(images[-1])
  return images


class Detection:
  def __init__(self, line):
    splitted = line.split()
    self.x1 = float(splitted[0])
    self.y1 = float(splitted[1])
    self.x2 = float(splitted[2])
    self.y2 = float(splitted[3])
    self.score = float(splitted[4])
    self.class_id = int(splitted[5])
    self.class_name = ' '.join(splitted[6:])


class Groundtruth:
  def __init__(self, line):
    splitted = line.split()
    self.class_name = splitted[0]
    self.x1 = float(splitted[4])
    self.y1 = float(splitted[5])
    self.x2 = float(splitted[6])
    self.y2 = float(splitted[7])


def filename_to_id(file_name, dataset_type):
  '''
  Returns identitifer of image in dataset.

  Each dataset has its own way how to identify
  particular image in detection results or annotations.
  '''
  short_name = os.path.splitext(file_name)[0]

  # In KITTI dataset image identifies by its name
  if dataset_type == KITTI:
    return int(short_name)

  # In COCO dataset ID is a number which is a part of filename
  if dataset_type == COCO:
    # COCO 2017: 000000000139.jpg
    # COCO 2014: COCO_val2014_000000000042.jpg
    if short_name[0] == '0':
      return int(short_name)
    else:
      return int(re.split(r'_', short_name)[2])

  # In OpenImages dataset image identifies by its name
  if dataset_type == OID:
    return short_name

  raise ValueError('Unknown datase type {}'.format(dataset_type))  


def print_header(s):
  print('\n' + '*'*80)
  print('* ' + s)
  print('*'*80)
