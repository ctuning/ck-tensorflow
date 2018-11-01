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

def prepare_dir(dir_path):
  '''
  Recreate a directory
  '''
  if os.path.isdir(dir_path):
    shutil.rmtree(dir_path)
  os.mkdir(dir_path)


def load_image_list(images_dir, images_count, skip_images):
  '''
  Load list of images to be processed
  '''
  assert os.path.isdir(images_dir), 'Input dir does not exit'
  files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
  files = [f for f in files if re.search(r'\.jpg$', f, re.IGNORECASE)
                            or re.search(r'\.jpeg$', f, re.IGNORECASE)]
  assert len(files) > 0, 'Input dir does not contain image files'
  files = sorted(files)[skip_images:]
  assert len(files) > 0, 'Input dir does not contain more files'
  images = files[:images_count]
  # Repeat last image to make full last batch
  if len(images) < images_count:
    for _ in range(images_count-len(images)):
      images.append(images[-1])
  return images
