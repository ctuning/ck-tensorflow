#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
from enum import Enum
import numpy as np

# Module parameters
# Assign externally before usage
LABELS_DIR = None 
CLASS_NAMES = None
DETECTION_IOU = None
FP_THRESHOLD = None


class Difficulty(Enum):
  IGNORE = -1
  EASY = 0
  MODERATE = 1
  HARD = 2

# Minimum height for evaluated groundtruth/detections
MIN_HEIGHT = {
  Difficulty.EASY: 40,
  Difficulty.MODERATE: 25,
  Difficulty.HARD: 10
}

# Maximum occlusion level of the groundtruth used for evaluation
MAX_OCCLUSION = {
  Difficulty.EASY: 0,
  Difficulty.MODERATE: 1,
  Difficulty.HARD: 2
}

# Maximum truncation level of the groundtruth used for evaluation
MAX_TRUNCATION = {
  Difficulty.EASY: 0.15,
  Difficulty.MODERATE: 0.3,
  Difficulty.HARD: 0.5
} 


class ImageResult(object):
  def __init__(self):
    self.image_file = '' # Processed image file without path
    self.label_file = '' # Labels file name without path
    self.detections = [] # Recognized list of DetectionBox


class BoxBase(object):
  def __init__(self):
    self.class_index = -1
    self.class_name = ''
    self.box = [0, 0, 0, 0]
    self.assigned = False

  def xmin(self): return self.box[0]
  def ymin(self): return self.box[1]
  def xmax(self): return self.box[2]
  def ymax(self): return self.box[3]
  def height(self): return self.ymax() - self.ymin()
  def width(self): return self.xmax() - self.xmin()
  def area(self): return self.width() * self.height()

  def difficulty(self):
    if not self.should_ignore(Difficulty.EASY):
      return Difficulty.EASY
    if not self.should_ignore(Difficulty.MODERATE):
      return Difficulty.MODERATE
    if not self.should_ignore(Difficulty.HARD):
      return Difficulty.HARD
    return Difficulty.IGNORE


class DetectionBox(BoxBase):
  def __init__(self):
    super(DetectionBox, self).__init__()
    self.prob = 0

  def str(self):
    return '{} ({}): {} {}'.format(self.class_name, self.class_index, self.prob, self.box)

  def should_ignore(self, difficulty):
    if self.height() < MIN_HEIGHT[difficulty]: return True
    return False


class GroundTruthBox(BoxBase):
  def __init__(self):
    super(GroundTruthBox, self).__init__()
    self.truncation = 0
    self.occlusion = 0

  def should_ignore(self, difficulty):
    '''
    Ground truth is ignored, if occlusion, truncation exceeds the difficulty
    or size is too small. It doesn't count as FN nor TP.
    '''
    if self.occlusion > MAX_OCCLUSION[difficulty]: return True
    if self.truncation > MAX_TRUNCATION[difficulty]: return True
    if self.height() < MIN_HEIGHT[difficulty]: return True
    return False


class Stat(object):
  def __init__(self):
    self.tp = 0 # true positive
    self.fp = 0 # false positive
    self.fn = 0 # false negative

  def add(self, stat):
    self.tp += stat.tp
    self.fp += stat.fp
    self.fn += stat.fn

  def str(self):
    return 'tp={}, fp={}, fn={}'.format(self.tp, self.fp, self.fn)


class ClassStat(object):
  def __init__(self):
    self.easy = None
    self.moderate = None
    self.hard = None
    self.count_easy = 0
    self.count_moderate = 0
    self.count_hard = 0

  def add(self, stat):
    if stat.easy:
      if not self.easy:
        self.easy = Stat()
      self.easy.add(stat.easy)
      self.count_easy += 1

    if stat.moderate:
      if not self.moderate:
        self.moderate = Stat()
      self.moderate.add(stat.moderate)
      self.count_moderate += 1

    if stat.hard:
      if not self.hard:
        self.hard = Stat()
      self.hard.add(stat.hard)
      self.count_hard += 1

  def str(self):
    return 'E ({}): {} | M ({}): {} | H ({}): {}'.format(
      self.count_easy, self.easy.str() if self.easy else 'None',
      self.count_moderate, self.moderate.str() if self.moderate else 'None',
      self.count_hard, self.hard.str() if self.hard else 'None'
    )


def load_groundtruth(label_file):
  groundtruth = []
  dontcare = []
  with open(os.path.join(LABELS_DIR, label_file), 'r') as f:
    for line in f:
      parts = line.split()
      assert len(parts) == 15
      class_name = parts[0].lower()
      if class_name in CLASS_NAMES:
        b = GroundTruthBox()
        b.class_index = CLASS_NAMES.index(class_name)
        b.class_name = class_name
        b.truncation = float(parts[1])
        b.occlusion = float(parts[2])
        b.box = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
        groundtruth.append(b)
      elif class_name == 'dontcare':
        # Overlapping with these boxes are ignored and not counted as false positive
        b = GroundTruthBox()
        b.box = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
        dontcare.append(b)
      else:
        # Ignore unknown class
        continue
  return groundtruth, dontcare


def box_overlap(a, b):
  w = min(a.xmax(), b.xmax()) - max(a.xmin(), b.xmin())
  h = min(a.ymax(), b.ymax()) - max(a.ymin(), b.ymin())
  if w <= 0 or h <= 0:
    return 0
  inter = w*h
  return inter / (a.area() + b.area() - inter)


def evaluate_difficulty(difficulty, detected, expected):
  expected = [e for e in expected if not e.assigned and not e.should_ignore(difficulty)]
  if not expected: return None
  
  stat = Stat()

  for d in detected:
    if d.assigned: continue
    
    for e in expected:
      overlap = box_overlap(d, e)
      if overlap >= DETECTION_IOU:
        # Ignore repeated detections for the same ground truth
        if not e.assigned:
          stat.tp += 1
          e.assigned = True
        d.assigned = True

  for e in expected:
    if not e.assigned:
      e.assigned = True
      stat.fn += 1

  return stat


def evaluate_class(class_name, all_detected, all_expected, dontcare):
  stat = ClassStat()
  detected = [d for d in all_detected if d.class_name == class_name]
  expected = [e for e in all_expected if e.class_name == class_name]
  stat.easy = evaluate_difficulty(Difficulty.EASY, detected, expected)
  stat.moderate = evaluate_difficulty(Difficulty.MODERATE, detected, expected)
  stat.hard = evaluate_difficulty(Difficulty.HARD, detected, expected)

  # Remainig unassigned boxes my be false-positive 
  for d in detected:
    if not d.assigned:
      is_fp = True
      # Ignore intersections with stuff areas,
      # do not count them as false-positive
      for stuff in dontcare:
        if box_overlap(d, stuff) >= DETECTION_IOU:
          is_fp = False
          break
      # Ignore some boxes by prob threshold
      if d.prob < FP_THRESHOLD:
        is_fp = False
      # Suggest difficulty for this detection
      if is_fp:
        difficulty = d.difficulty()
        if difficulty == Difficulty.EASY:
          if not stat.easy:
            stat.easy = Stat()
          stat.easy.fp += 1
        elif difficulty == Difficulty.MODERATE:
          if not stat.moderate:
            stat.moderate = Stat()
          stat.moderate.fp += 1
        elif difficulty == Difficulty.HARD:
          if not stat.hard:
            stat.hard = Stat()
          stat.hard.fp += 1
  return stat


def evaluate_image(detections, label_file, all_stat):
  groundtruth, dontcare = load_groundtruth(label_file)
  for class_name in CLASS_NAMES:
    stat = evaluate_class(class_name, detections, groundtruth, dontcare)
    all_stat[class_name].add(stat)


def evaluate_images(images_results, results_info, classes_metrics):
  all_stat = {}
  for class_name in CLASS_NAMES:
    all_stat[class_name] = ClassStat()

  # Accumulate statistics for all images
  for res in images_results:
    evaluate_image(res.detections, res.label_file, all_stat)

  # Calculate averaged statistics
  stat_count = 0
  recalls = []
  precisions = []

  # Calculate separate statistics for each difficulty level
  def calc_metrics(stat, difficulty_name, result_AP, result_recall):
    if not stat:
      result_AP[difficulty_name] = 0
      result_recall[difficulty_name] = 0
      print('{}: None'.format(difficulty_name))
      return
    recall = (stat.tp / float(stat.tp + stat.fn)) if stat.tp + stat.fn > 0 else 0
    precision = (stat.tp / float(stat.tp + stat.fp)) if stat.tp + stat.fp > 0 else 0
    recalls.append(recall)
    precisions.append(precision)
    result_AP[difficulty_name] = precision
    result_recall[difficulty_name] = recall
    print('{}: recall={:.3f}, AP={:.3f}'.format(difficulty_name, recall, precision))

  # Calculate separate statistics for each class
  for class_name in CLASS_NAMES:
    print('\n{}'.format(class_name.upper()))
    stat = all_stat[class_name]
    result_AP = {}
    result_recall = {}
    calc_metrics(stat.easy, 'easy', result_AP, result_recall)
    calc_metrics(stat.moderate, 'moderate', result_AP, result_recall)
    calc_metrics(stat.hard, 'hard', result_AP, result_recall)
    stat_count += stat.count_easy + stat.count_moderate + stat.count_hard
    classes_metrics[class_name]['AP'] = result_AP
    classes_metrics[class_name]['recall'] = result_recall

  print('\nSummary:')
  for class_name in CLASS_NAMES:
    print('{}: {}'.format(class_name, all_stat[class_name].str()))
  print('stat_count={}'.format(stat_count))

  recall = np.sum(recalls) / stat_count
  mAP = np.sum(precisions) / stat_count
  print('')
  print('recall = {:.3f}'.format(recall))
  print('mAP = {:.3f}'.format(mAP))

  results_info['recall'] = recall
  results_info['mAP'] = mAP
