#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import re

KITTI = 'kitti'
COCO = 'coco'

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
    return int(re.split(r'_', short_name)[2])

  raise ValueError('Unknown datase type {}'.format(dataset_type))
