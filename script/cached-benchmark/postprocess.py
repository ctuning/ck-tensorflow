#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import json

TOP1 = 0
TOP5 = 0
IMAGES_COUNT = 0 # to be assigned

def ck_postprocess(i):
  print('\n--------------------------------')
  def my_env(var): return i['env'].get(var)
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)

  # Init variables from environment
  BATCH_COUNT = int(my_env('CK_BATCH_COUNT'))
  BATCH_SIZE = int(my_env('CK_BATCH_SIZE'))
  global IMAGES_COUNT
  IMAGES_COUNT = BATCH_COUNT * BATCH_SIZE
  SKIP_IMAGES = int(my_env('CK_SKIP_IMAGES'))
  RESULTS_DIR = 'predictions'
  NUM_CLASSES = 1000
  AUX_DIR = dep_env('imagenet-aux', 'CK_ENV_DATASET_IMAGENET_AUX')
  CLASSES_FILE = os.path.join(AUX_DIR, 'synset_words.txt')
  VALUES_FILE = os.path.join(AUX_DIR, 'val.txt')
  CLASSES_LIST = []
  VALUES_MAP = {}
  IMAGE_FILE = my_env('CK_IMAGE_FILE')
  FULL_REPORT = my_env('CK_SILENT_MODE') != 'YES'


  # Loads ImageNet classes and correct predictions
  def load_ImageNet_classes():
    classes_list = []  
    with open(CLASSES_FILE, 'r') as classes_file:
      classes_list = classes_file.read().splitlines()

    values_map = {}
    with open(VALUES_FILE, 'r') as values_file:
      if IMAGE_FILE:
        # Single file mode: try to find this file in values
        for line in values_file:
          file_name, file_class = line.split()
          if file_name == IMAGE_FILE:
            values_map[file_name] = int(file_class)
            break
      else:
        # Directory mode: load only required amount of values
        for _ in range(SKIP_IMAGES):
          values_file.readline().split()
        for _ in range(IMAGES_COUNT):
          val = values_file.readline().split()
          values_map[val[0]] = int(val[1])

    return classes_list, values_map


  # Returns printable string for ImageNet specific class
  def get_class_str(class_index):
    obj_class = CLASSES_LIST[class_index]
    if len(obj_class) > 50:
        obj_class = obj_class[:50] + '...'
    return '(%d) %s' % (class_index, obj_class)


  # Shows prediction results for image file
  # top5 - list of pairs (prob, class_index)
  def print_predictions(top5, img_file):
    print('---------------------------------------')
    if img_file in VALUES_MAP:
      class_correct = VALUES_MAP[img_file]
      print('%s - %s' % (img_file, get_class_str(class_correct)))
    else:
      print(img_file)
    for prob, class_index in top5:
      print('%.2f - %s' % (prob, get_class_str(class_index)))
    print('---------------------------------------')       


  # Returns list of pairs (prob, class_index)
  def get_top5(all_probs):
    probs_with_classes = []
    for class_index in range(len(all_probs)):
      prob = all_probs[class_index]
      probs_with_classes.append((prob, class_index))
    sorted_probs = sorted(probs_with_classes, key = lambda pair: pair[0], reverse=True)
    return sorted_probs[0:5]    


  # Calculates if prediction was correct for specified image file
  # top5 - list of pairs (prob, class_index)
  def check_predictions(top5, img_file):
    if img_file not in VALUES_MAP:
      print('Correctness information is not available')
      return {}

    class_correct = VALUES_MAP[img_file]
    classes = [c[1] for c in top5]
    is_top1 = class_correct == classes[0]
    is_top5 = class_correct in classes
    if is_top1:
      global TOP1
      TOP1 += 1
    if is_top5:
      global TOP5
      TOP5 += 1
    return {
      'accuracy_top1': 'yes' if is_top1 else 'no',
      'accuracy_top5': 'yes' if is_top5 else 'no',
      'class_correct': class_correct,
      'class_topmost': classes[0],
      'file_name': img_file
    }

  frame_predictions = []

  def calculate_precision():
    print('Process results in {}'.format(RESULTS_DIR))

    def load_probes(filename):
      probes = []
      with open(os.path.join(RESULTS_DIR, filename), 'r') as f:
        for line in f:
          s = line.strip()
          if s: probes.append(float(s))
      return probes 

    checked_files = 0

    for res_file in sorted(os.listdir(RESULTS_DIR)):
      # remove trailing suffix .txt
      img_file = res_file[:-4] 
      checked_files += 1
      
      all_probes = load_probes(res_file)
      if len(all_probes) != NUM_CLASSES:
        print('WARNING: {} is invalid probes count in file {}, results ignored'.format(len(all_probes), res_file))
        global IMAGES_COUNT
        IMAGES_COUNT -= 1
        continue
        
      top5 = get_top5(all_probes)
      if FULL_REPORT:
        print_predictions(top5, img_file)
      elif checked_files % 100 == 0:
        print('Predictions checked: {}'.format(checked_files))
      res = check_predictions(top5, img_file)
      frame_predictions.append(res)


  global TOP1
  global TOP5
  TOP1 = 0
  TOP5 = 0
  CLASSES_LIST, VALUES_MAP = load_ImageNet_classes()
  calculate_precision()

  accuracy_top1 = TOP1 / float(IMAGES_COUNT) if IMAGES_COUNT > 0 else 0
  accuracy_top5 = TOP5 / float(IMAGES_COUNT) if IMAGES_COUNT > 0 else 0 

  # Store benchmark results
  openme = {}
  
  # Preserve values stored by program
  with open('tmp-ck-timer.json', 'r') as o:
    old_values = json.load(o)
  for key in old_values:
    # xopenmp c++ writes this section, copy it into root object
    if key == 'run_time_state':
      for key1 in old_values[key]:
        openme[key1] = old_values[key][key1]
    else:
      openme[key] = old_values[key]

  setup_time = openme.get('setup_time_s', 0.0)
  test_time = openme.get('test_time_s', 0.0)
  total_load_images_time = openme.get('images_load_time_total_s', 0.0)
  total_prediction_time = openme.get('prediction_time_total_s', 0.0)
  avg_prediction_time = openme.get('prediction_time_avg_s', 0.0)

  # Print metrics
  print('\nSummary:')
  print('-------------------------------')
  print('Graph loaded in {:.6f}s'.format(setup_time))
  print('All images loaded in {:.6f}s'.format(total_load_images_time))
  print('All images classified in {:.6f}s'.format(total_prediction_time))
  print('Average classification time: {:.6f}s'.format(avg_prediction_time))
  print('Accuracy top 1: {} ({} of {})'.format(accuracy_top1, TOP1, IMAGES_COUNT))
  print('Accuracy top 5: {} ({} of {})'.format(accuracy_top5, TOP5, IMAGES_COUNT))  

  openme['accuracy_top1'] = accuracy_top1
  openme['accuracy_top5'] = accuracy_top5
  openme['frame_predictions'] = frame_predictions
  openme['execution_time'] = total_prediction_time
  openme['execution_time_sum'] = setup_time + test_time

  with open('tmp-ck-timer.json', 'w') as o:
    json.dump(openme, o, indent=2, sort_keys=True)

  print('--------------------------------\n')
  return {'return': 0}  

