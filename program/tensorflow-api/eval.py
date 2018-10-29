import sys
import os
sys.path.append(os.getenv("CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR"))

import time
import json
import shutil
import numpy as np
import tensorflow as tf
import metriconv
import metricstat
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import ops as utils_ops

CUR_DIR=os.getcwd()
PATH_TO_FROZEN_GRAPH = os.getenv("CK_ENV_MODEL_TENSORFLOW_API_FROZEN_GRAPH")
print("Frozen graph: "+  PATH_TO_FROZEN_GRAPH)
PATH_TO_CATEGORY_LABELS = os.getenv("CK_ENV_MODEL_TENSORFLOW_API_LABELS")
print("Category labels: " + PATH_TO_CATEGORY_LABELS)
IMAGES_DIR = os.getenv("CK_ENV_DATASET_IMAGE_DIR")
print("Images directory: " + IMAGES_DIR)
SAVE_IMAGES = os.getenv("CK_SAVE_IMAGES") == "YES"
if SAVE_IMAGES:
  IMAGES_OUT_DIR=os.path.join(CUR_DIR,"out-images")
  print("Images output directory: " + IMAGES_OUT_DIR)
LABELS_OUT_DIR=os.path.join(CUR_DIR,"out-labels")
print("Labels output directory: " + LABELS_OUT_DIR)
DATASET_TYPE = os.getenv("CK_ENV_DATASET_TYPE")
if DATASET_TYPE:
  print("Dataset type: " + DATASET_TYPE)
else:
  DATASET_TYPE = "coco"
  print("Dataset type not setted. Default settings is: "  + DATASET_TYPE)
MODEL_DATASET_TYPE = os.getenv("CK_ENV_MODEL_DATASET_TYPE")
if MODEL_DATASET_TYPE:
  print("Model's dataset type: " + MODEL_DATASET_TYPE)
else:
  MODEL_DATASET_TYPE = "coco"
  print("Model's dataset type not setted. Default settings is: "  + MODEL_DATASET_TYPE)
TARGET_METRIC_TYPE = os.getenv("CK_ENV_TARGET_METRIC_TYPE")
if TARGET_METRIC_TYPE:
  print("Target metric type: " + TARGET_METRIC_TYPE)
else:
  TARGET_METRIC_TYPE = DATASET_TYPE
  print("Target metric type isn't defined. Default is equal to "\
    "dataset's type ("  + TARGET_METRIC_TYPE + ")")
DATASET_ANNOTATIONS = os.getenv("CK_ENV_DATASET_ANNOTATIONS")
print("Dataset annotations: " + DATASET_ANNOTATIONS)

print('*'*80)

TARGET_ANNOTATIONS_DIR=os.path.join(CUR_DIR,"annotations")
if TARGET_METRIC_TYPE != DATASET_TYPE:
  if os.path.isdir(TARGET_ANNOTATIONS_DIR):
    shutil.rmtree(TARGET_ANNOTATIONS_DIR)
  os.mkdir(TARGET_ANNOTATIONS_DIR)

TARGET_RESULTS_DIR=os.path.join(CUR_DIR,"results")
if os.path.isdir(TARGET_RESULTS_DIR):
  shutil.rmtree(TARGET_RESULTS_DIR)
os.mkdir(TARGET_RESULTS_DIR)


if SAVE_IMAGES:
  if os.path.isdir(IMAGES_OUT_DIR):
    shutil.rmtree(IMAGES_OUT_DIR)
  os.mkdir(IMAGES_OUT_DIR)

if os.path.isdir(LABELS_OUT_DIR):
  shutil.rmtree(LABELS_OUT_DIR)
os.mkdir(LABELS_OUT_DIR)

setup_time_begin = time.time()

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_CATEGORY_LABELS, use_display_name=True)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  begin_time = time.time()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
  weights_load_time = time.time() - begin_time
  print('Weights loaded in {:.4f}s\n'.format(weights_load_time))

setup_time = time.time() - setup_time_begin

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

IMAGE_FILES=[f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

file_counter = 0
test_time_begin = time.time()
detect_time_total = 0
load_time_total = 0
print('-'*80)

processed_images = []
for image_file in IMAGE_FILES:
  file_counter += 1
  if file_counter > 3 :
    break
  print('\n'+image_file + ': ' + `file_counter` + ' of ' + `len(IMAGE_FILES)`)
  load_time_begin = process_time_begin = time.time()
  image = Image.open(os.path.join(IMAGES_DIR, image_file))
  image_id = metriconv.filename_to_id(image_file, DATASET_TYPE)
  processed_images.append(image_id)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  load_time = time.time() - load_time_begin
  load_time_total += load_time
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  # image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  detect_time_begin = time.time()
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  detect_time = time.time() - detect_time_begin
  detect_time_total += detect_time
  # Visualization of the results of a detection.
  # Process results
  (im_width, im_height) = image.size
  file_name = os.path.splitext(image_file)[0]
  res_file = os.path.join(LABELS_OUT_DIR, file_name) + '.txt'
  with open(res_file, 'w') as f:
    f.write('{:d} {:d}\n'.format(im_width, im_height))
    for i in range(output_dict['num_detections']):
      class_id = output_dict['detection_classes'][i]
      if 'display_name' in category_index[class_id]:
        class_name = category_index[class_id]['display_name']
      else:
        class_name = category_index[class_id]['name']
      y1, x1, y2, x2 = output_dict['detection_boxes'][i]
      score = output_dict['detection_scores'][i]
      f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:d} {}\n'.format( x1*im_width,
        y1*im_height, x2*im_width, y2*im_height, score, class_id, class_name))
#      f.write('{} -1 -1 0.0 {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0 0.0 0.0'\
#        ' 0.0 0.0 0.0 {:.3f}\n'.format(class_name, y1*im_width, x1*im_height,
#        y2*im_width, x2*im_height, score))

  if SAVE_IMAGES:
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=2)
    image=Image.fromarray(image_np)
    image.save(os.path.join(IMAGES_OUT_DIR, image_file))
  process_time = time.time() - process_time_begin
  print('  Full processing time: {:.4f}s'.format(process_time))
  print('             Load time: {:.4f}s'.format(load_time))
  print('        Detecting time: {:.4f}s'.format(detect_time))

print('-'*80)
test_time = time.time() - test_time_begin
detect_avg_time = detect_time_total / len(IMAGE_FILES)
load_avg_time = load_time_total / len(IMAGE_FILES)

# Store benchmark results
openme = {}
openme['setup_time_s'] = setup_time
openme['test_time_s'] = test_time
openme['weights_load_time_s'] = weights_load_time
openme['images_load_time_s'] = load_time_total
openme['images_load_time_avg_s'] = load_avg_time
openme['detection_time_total_s'] = detect_time_total
openme['detection_time_avg_s'] = detect_avg_time

openme['avg_time_ms'] = detect_avg_time * 1000
openme['avg_fps'] = 1.0 / detect_avg_time if detect_avg_time > 0 else 0
openme['batch_time_ms'] = detect_time_total * 1000
openme['batch_size'] = len(IMAGE_FILES)

with open('tmp-ck-timer.json', 'w') as o:
  json.dump(openme, o, indent=2, sort_keys=True)

with open('processed_images_id.json', 'w') as wf:
  wf.write(json.dumps(processed_images))

print('*'*80)
print('* Postprocess results')
print('*'*80)
print('\n Process annotations...')
annotations = metriconv.convert_annotations(DATASET_ANNOTATIONS, TARGET_ANNOTATIONS_DIR, DATASET_TYPE , TARGET_METRIC_TYPE)
if not annotations:
  print('Error converting annotations from ' + DATASET_TYPE + ' to ' + TARGET_METRIC_TYPE)
  sys.exit()

print('\n Converting results...')
results = metriconv.convert_results(LABELS_OUT_DIR, TARGET_RESULTS_DIR, DATASET_TYPE,
    MODEL_DATASET_TYPE , TARGET_METRIC_TYPE)
if not results:
  print('Error converting results from type ' + MODEL_DATASET_TYPE + ' to ' + TARGET_METRIC_TYPE)

print('\n Evaluating results...')
eval_res = metricstat.evaluate(processed_images, results, annotations, TARGET_METRIC_TYPE)
