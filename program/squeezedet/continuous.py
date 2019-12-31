# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

# Original license text is below
# BSD 2-Clause License
#
# Copyright (c) 2016, Bichen Wu
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from threading import Thread

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from nets import *
from utils.util import bbox_transform

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'image_dir', './', """Directory with images""")
tf.app.flags.DEFINE_string(
    'label_dir', './', """Directory with image labels""")
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.7, """Threshold for IoU metric to determine false positives""")
tf.app.flags.DEFINE_integer(
    'gpu', 0, """GPU ID""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")
tf.app.flags.DEFINE_string(
    'finisher_file', '', """Finisher file. If present, the app stops. Useful to interrupt the continuous mode.""")
tf.app.flags.DEFINE_integer(
    'input_device', -1, """Input device (like webcam) ID. If specified, images are taken from this device instead of image dir.""")
tf.app.flags.DEFINE_integer(
    "webcam_max_image_count", 10000, "Maximum image count generated in the webcam mode.");
tf.app.flags.DEFINE_string(
    "skip_files_including", "", "Skip files from the beginning to the given one (inclusive)");

UNASSIGNED = -2
UNKNOWN = -1
EASY = 0
MODERATE = 1
HARD = 2
MIN_HEIGHT     = [40, 25, 25]        # minimum height for evaluated groundtruth/detections
MAX_OCCLUSION  = [0, 1, 2]           # maximum occlusion level of the groundtruth used for evaluation
MAX_TRUNCATION = [0.15, 0.3, 0.5] # maximum truncation level of the groundtruth used for evaluation

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def my_draw_box(im, box_list, label_list, color=(0,255,0), cdict=None, label_placement='bottom'):
    assert label_placement == 'bottom' or label_placement == 'top', \
        'label_placement format not accepted: {}.'.format(label_placement)

    for bbox, label in zip(box_list, label_list):

        xmin, ymin, xmax, ymax = [int(b) for b in bbox]

        l = label.split(':')[0] # text before "CLASS: (PROB)"
        if cdict and l in cdict:
            c = cdict[l]
        else:
            c = color

        # draw box
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
        # draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        if label_placement == 'bottom':
            cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)
        else:
            cv2.putText(im, label, (xmin, ymin), font, 0.3, c, 1)

box_counter = 0

class Box:
    def __init__(self, klass, bbox, occlusion=0, truncation=0, prob=0):
        global box_counter
        self.id = box_counter
        box_counter += 1
        self.klass = klass
        self.bbox = bbox
        self.occlusion = occlusion
        self.truncation = truncation
        self.prob = prob
        self.assigned_difficulty = UNASSIGNED

    def height(self):
        return self.bbox[3] - self.bbox[1]

    def should_ignore(self, difficulty):
        return self.occlusion > MAX_OCCLUSION[difficulty] or self.truncation > MAX_TRUNCATION[difficulty] or self.height() < MIN_HEIGHT[difficulty]

    def difficulty(self):
        if not self.should_ignore(EASY):
            return EASY
        if not self.should_ignore(MODERATE):
            return MODERATE
        if not self.should_ignore(HARD):
            return HARD
        return UNKNOWN

def care(r_box, dontcare):
    threshold = 0.7 if 'car' == r_box.klass else 0.5
    for dc_box in dontcare:
        iou = bb_intersection_over_union(r_box.bbox, dc_box.bbox)
        if iou >= threshold:
            return False
    return True

def eval_boxes(expected, recognized, klass, difficulty):
    gt = [b for b in expected if b.klass == klass and difficulty == b.difficulty()]
    rec = []
    if UNKNOWN == difficulty:
        rec = [b for b in recognized if b.klass == klass]
    else:
        rec = [b for b in recognized if b.klass == klass and UNASSIGNED == b.assigned_difficulty and b.height() >= MIN_HEIGHT[difficulty]]
    assigned_rec = [False for b in rec]
    assigned_gt = [False for b in gt]
    tp = 0
    fn = 0
    for r_index, r_box in enumerate(rec):
        threshold = 0.7 if 'car' == r_box.klass else 0.5

        for gt_index, gt_box in enumerate(gt):
            if assigned_gt[gt_index]:
                continue
            iou = bb_intersection_over_union(r_box.bbox, gt_box.bbox)
            # print('+ r_id {}, gt_id {}, iou {}'.format(r_box.id, gt_box.id, iou))
            if iou >= threshold:
                assigned_rec[r_index] = True
                assigned_gt[gt_index] = True
                r_box.assigned_difficulty = difficulty
                break

        if assigned_rec[r_index]:
            tp += 1
        else:
            fn += 1

    return (tp, len(gt))

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class Stat:
    def __init__(self):
        self.avg = 0.0
        self.count = 0

    def add(self, v):
        self.count += 1
        self.avg = (self.avg * (self.count - 1) + v) / float(self.count)

    def addIf(self, v, condition):
        if condition or not isclose(v, 0):
            self.add(v)

def calc_mAP(avg_precision):
    s = 0.0
    count = 0
    for v in avg_precision.values():
        for stat in v:
            if 0 < stat.count:
                s += stat.avg
                count += 1
    return 0 if 0 == count else s / float(count)

def rescale(x, orig_scale, target_scale):
    return float(target_scale) * (float(x) / float(orig_scale))

def rescale_boxes(boxes, boxes_shape, target_shape):
    bh, bw = boxes_shape[:2]
    th, tw = target_shape[:2]
    ret = []
    for box in boxes:
        b = box.bbox
        nb = [rescale(b[0], bw, tw), rescale(b[1], bh, th), rescale(b[2], bw, tw), rescale(b[3], bh, th)]
        ret.append(Box(box.klass, nb, occlusion=box.occlusion, truncation=box.truncation, prob=box.prob))
    return ret

def safe_div(all_rec, tp):
    return 0 if 0 == all_rec else (float(tp) / float(all_rec))

def detect_image(mc, sess, model, class_names, avg_precision, orig_im, file_name, original_file_path):
    global box_counter
    box_counter = 0

    if os.environ.get('CODEREEF','')=='YES':
      try:
        boxed_img = orig_im.copy()
      except:
        return

    else:
        boxed_img = orig_im.copy()

    im = orig_im.astype(np.float32, copy=True)
    im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
    input_image = im - mc.BGR_MEANS

    start_clock = time.time()

    # Detect
    det_boxes, det_probs, det_class = sess.run(
        [model.det_boxes, model.det_probs, model.det_class],
        feed_dict={model.image_input:[input_image]})

    # Filter
    final_boxes, final_probs, final_class = model.filter_prediction(det_boxes[0], det_probs[0], det_class[0])

    duration = time.time() - start_clock

    keep_idx    = [idx for idx in range(len(final_probs)) \
                      if final_probs[idx] > mc.PLOT_PROB_THRESH]
    final_boxes = [final_boxes[idx] for idx in keep_idx]
    final_probs = [final_probs[idx] for idx in keep_idx]
    final_class = [final_class[idx] for idx in keep_idx]

    recognized = [Box(class_names[k], bbox_transform(bbox), prob=p) for k, bbox, p in zip(final_class, final_boxes, final_probs)]

    # TODO(bichen): move this color dict to configuration file
    cls2clr = {
        'car': (255, 191, 0),
        'cyclist': (0, 191, 255),
        'pedestrian':(255, 0, 191)
    }

    expected = []
    dontcare = []
    class_count = dict((k, 0) for k in class_names)
    
    if FLAGS.label_dir:
        label_file_name = os.path.join(FLAGS.label_dir, file_name)
        label_file_name = os.path.splitext(label_file_name)[0] + '.txt'
        if os.path.isfile(label_file_name):
            with open(label_file_name) as lf:
                label_lines = [x.strip() for x in lf.readlines()]
                for l in label_lines:
                    parts = l.strip().lower().split(' ')
                    klass = parts[0]
                    bbox = [float(parts[i]) for i in [4, 5, 6, 7]]
                    if klass in class_count.keys():
                        class_count[klass] += 1
                        b = Box(klass, bbox, truncation=float(parts[1]), occlusion=float(parts[2]))
                        expected.append(b)
                    elif klass == 'dontcare':
                        dontcare.append(Box(klass, bbox))

    expected_class_count = class_count

    rescaled_recognized = rescale_boxes(recognized, im.shape, orig_im.shape)

    # Draw dontcare boxes
    my_draw_box(
        boxed_img, [b.bbox for b in dontcare],
        ['dontcare' for b in dontcare],
        label_placement='top', color=(255,255,255)
    )
    # Draw original boxes
    my_draw_box(
        boxed_img, [b.bbox for b in expected],
        [box.klass + ': (TRUE)' for box in expected],
        label_placement='top', color=(200,200,200)
    )
    # Draw recognized boxes
    my_draw_box(
        boxed_img, [b.bbox for b in rescaled_recognized],
        [b.klass + ': (%.2f)' % b.prob for b in rescaled_recognized],
        cdict=cls2clr,
    )

    out_file_name = os.path.join(FLAGS.out_dir, file_name)
    cv2.imwrite(out_file_name, orig_im)

    if os.environ.get('CODEREEF','')!='YES':
       boxed_out_file_name = os.path.join(FLAGS.out_dir, 'boxed_' + file_name)
       cv2.imwrite(boxed_out_file_name, boxed_img)

    results={'objects':[]}
    
    print('File: {}'.format(out_file_name))
    if '' != original_file_path:
        print('Original file: {}'.format(original_file_path))
    print('Duration: {} sec'.format(duration))

    class_count = dict((k, 0) for k in class_names)
    for k in final_class:
        class_count[class_names[k]] += 1

    for k, v in class_count.items():
        print('Recognized {}: {}'.format(k, v))

    for k, v in expected_class_count.items():
        print('Expected {}: {}'.format(k, v))

    for box in rescaled_recognized:
        b = box.bbox
        print('Detection {}: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(box.klass, b[0], b[1], b[2], b[3], box.prob))

        if os.environ.get('CODEREEF','')=='YES':
           obj={'name':box.klass,
                'x':int(b[0]),
                'y':int(b[1]),
                'w':int(b[2])-int(b[0]),
                'h':int(b[3])-int(b[1]),
                'probability':float(box.prob)}
           results['objects'].append(obj)

    for box in expected:
        b = box.bbox
        print('Ground truth {}: {:.3f} {:.3f} {:.3f} {:.3f} 1'.format(box.klass, b[0], b[1], b[2], b[3]))

    # Record JSON
    if os.environ.get('CODEREEF','')=='YES':

       if os.path.isfile(original_file_path):
          os.remove(original_file_path)

       codereef_out_file_name = os.path.join(FLAGS.out_dir, os.path.splitext(file_name)[0]+'.json')
       if not os.path.isfile(codereef_out_file_name):
          import json
          with open(codereef_out_file_name, 'w') as of:
               of.write(json.dumps(results, indent=2, sort_keys=True))

    expected = [b for b in expected if care(b, dontcare)]
    recognized = [b for b in recognized if care(b, dontcare)]
    for k in class_names:
        all_rec = len([b for b in recognized if b.klass == k])
        all_gt = len([b for b in expected if b.klass == k])

        report = 0 != all_rec or 0 != all_gt # don't report not found and actually unexpected labels, but still count them for mAP

        eval_boxes(expected, recognized, k, UNKNOWN)
        tp_easy, all_gt_easy = eval_boxes(expected, recognized, k, EASY)
        tp_mod, all_gt_mod = eval_boxes(expected, recognized, k, MODERATE)
        tp_hard, all_gt_hard = eval_boxes(expected, recognized, k, HARD)
        tp = tp_easy + tp_mod + tp_hard
        fp = all_rec - tp

        if report:
            print('True positive {}: {} easy, {} moderate, {} hard'.format(k, tp_easy, tp_mod, tp_hard))
            print('False positive {}: {}'.format(k, fp))

        precision = [
            safe_div(tp_easy + fp, tp_easy),
            safe_div(tp_mod + fp, tp_mod),
            safe_div(tp_hard + fp, tp_hard)
        ]

        recall = 0.0
        if 0 == all_gt:
            recall = 1.0 if 0 == all_rec else 0.0
        else:
            recall = float(tp) / float(all_gt)

        if report:
            print('Precision {}: {:.2f} easy, {:.2f} moderate, {:.2f} hard'.format(k, precision[EASY], precision[MODERATE], precision[HARD]))
            print('Recall {}: {:.2f}'.format(k, recall))

        ap = avg_precision[k]
        ap[EASY].addIf(precision[EASY], 0 < all_gt_easy)
        ap[MODERATE].addIf(precision[MODERATE], 0 < all_gt_mod)
        ap[HARD].addIf(precision[HARD], 0 < all_gt_hard)

        if report:
            print('Rolling AP {}: {:.2f} easy, {:.2f} moderate, {:.2f} hard'.format(k, ap[EASY].avg, ap[MODERATE].avg, ap[HARD].avg))

    print('Rolling mAP: {:.4f}'.format(calc_mAP(avg_precision)))
    print('')
    sys.stdout.flush()

def should_finish():
    return '' != FLAGS.finisher_file and os.path.isfile(FLAGS.finisher_file)

class WebcamVideoStream:
    """This class is a modified version of the class taken from the 'imutils' library

    The MIT License (MIT)

    Copyright (c) 2015-2016 Adrian Rosebrock, http://www.pyimagesearch.com

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    """
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return (self.grabbed, self.frame)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def detect_webcam(fn, device_id):
    vs = WebcamVideoStream(device_id).start()
    i = 0
    while not should_finish():
        ret, im = vs.read()
        if not ret:
            break
        fn(im, 'webcam_%06d.jpg' % i, '')
        i = (i + 1) % FLAGS.webcam_max_image_count
    vs.stop()

def detect_dir(fn, d):
    image_list = sorted([os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
    if '' != FLAGS.skip_files_including:
        try:
            i = image_list.index(FLAGS.skip_files_including)
            image_list = image_list[i+1:]
        except ValueError:
            pass
    for f in image_list:
        if should_finish():
            break
        im = cv2.imread(f)
        fn(im, os.path.split(f)[1], f)

def detect_dir_codereef(fn, d):
    image_list = sorted([os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
    if '' != FLAGS.skip_files_including:
        try:
            i = image_list.index(FLAGS.skip_files_including)
            image_list = image_list[i+1:]
        except ValueError:
            pass
    for f in image_list:
        if should_finish():
            break
        im = cv2.imread(f)
        fn(im, os.path.split(f)[1], f)

def image_demo():
  """Detect image."""

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+' or FLAGS.demo_net == 'resnet50' or FLAGS.demo_net == 'vgg16', \
    'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'resnet50':
      mc = kitti_res50_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = ResNet50ConvDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'vgg16':
      mc = kitti_vgg16_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = VGG16ConvDet(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      class_names = [k.lower() for k in mc.CLASS_NAMES]
      avg_precision = dict((k, [Stat(), Stat(), Stat()]) for k in class_names)

      fn = partial(detect_image, mc, sess, model, class_names, avg_precision)
      if 0 <= FLAGS.input_device:
        detect_webcam(fn, FLAGS.input_device)
      else:
        if os.environ.get('CODEREEF','')=='YES':
           while True:
              print ("Searching for images ...")
              detect_dir_codereef(fn, FLAGS.image_dir)
              import time
              time.sleep(0.1)
        else:
           detect_dir(fn, FLAGS.image_dir)

def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  image_demo()

if __name__ == '__main__':
    tf.app.run()
