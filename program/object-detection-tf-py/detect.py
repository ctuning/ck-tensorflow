#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import sys
import os
import json
import numpy as np
import shutil
import time
import cv2
import ck_utils

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

params = {}
params["CUR_DIR"] = os.getcwd()

# Model parameters
params["FROZEN_GRAPH"] = os.getenv("CK_ENV_TENSORFLOW_MODEL_FROZEN_GRAPH")
params["DEFAULT_WIDTH"] = int(os.getenv("CK_ENV_TENSORFLOW_MODEL_DEFAULT_WIDTH"))
params["DEFAULT_HEIGHT"] = int(os.getenv("CK_ENV_TENSORFLOW_MODEL_DEFAULT_HEIGHT"))
params["LABELMAP_FILE"] = os.getenv("CK_ENV_TENSORFLOW_MODEL_LABELMAP_FILE")
params["MODEL_DATASET_TYPE"] = os.getenv("CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE")

# Dataset parameters
params["IMAGES_DIR"] = os.getenv("CK_ENV_DATASET_IMAGE_DIR")
params["DATASET_TYPE"] = os.getenv("CK_ENV_DATASET_TYPE")
# Annotations can be a directory or a single file, depending on dataset type
params["ANNOTATIONS_PATH"] = os.getenv("CK_ENV_DATASET_ANNOTATIONS",'')

# Program Parameters 
params["CUSTOM_MODEL"] = int(os.getenv('CK_CUSTOM_MODEL', 0))
params["WITH_TENSORRT"] = int(os.getenv('CK_ENABLE_TENSORRT', 0))
params["TENSORRT_PRECISION"] = os.getenv('CK_TENSORRT_PRECISION', 'FP32')
params["TENSORRT_DYNAMIC"] = int(os.getenv('CK_TENSORRT_DYNAMIC', 0))
params["BATCH_COUNT"] = int(os.getenv('CK_BATCH_COUNT', 1))
params["BATCH_SIZE"] = int(os.getenv('CK_BATCH_SIZE', 1))
params["ENABLE_BATCH"] = int(os.getenv('CK_ENABLE_BATCH', 0))
params["RESIZE_WIDTH_SIZE"] = int(os.getenv('CK_ENV_IMAGE_WIDTH', params["DEFAULT_WIDTH"]))
params["RESIZE_HEIGHT_SIZE"] = int(os.getenv('CK_ENV_IMAGE_HEIGHT', params["DEFAULT_HEIGHT"]))
params["SKIP_IMAGES"] = int(os.getenv('CK_SKIP_IMAGES', 0))
params["SAVE_IMAGES"] = os.getenv("CK_SAVE_IMAGES") == "YES"
params["METRIC_TYPE"] = (os.getenv("CK_METRIC_TYPE") or params["DATASET_TYPE"]).lower()
params["IMAGES_OUT_DIR"] = os.path.join(params["CUR_DIR"], "images")
params["DETECTIONS_OUT_DIR"] = os.path.join(params["CUR_DIR"], "detections")
params["ANNOTATIONS_OUT_DIR"] = os.path.join(params["CUR_DIR"], "annotations")
params["RESULTS_OUT_DIR"] = os.path.join(params["CUR_DIR"], "results")
params["FULL_REPORT"] = os.getenv('CK_SILENT_MODE') == 'NO'
params["SKIP_DETECTION"] = os.getenv('CK_SKIP_DETECTION') == 'YES'
params["IMAGE_LIST_FILE"] = 'processed_images_id.json'
params["TIMER_JSON"] = 'tmp-ck-timer.json'
params["ENV_JSON"] = 'env.json'


def make_tf_config():
  mem_percent = float(os.getenv('CK_TF_GPU_MEMORY_PERCENT', 50))
  num_processors = int(os.getenv('CK_TF_CPU_NUM_OF_PROCESSORS', 0))

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.allocator_type = 'BFC'
  config.gpu_options.per_process_gpu_memory_fraction = mem_percent / 100.0
  if num_processors > 0:
    config.device_count["CPU"] = num_processors
  return config


##### function for "normal processing" (configuration = 0, no batch no custom no tensorRT)
def load_graph_traditional(params):

  graph_def = tf.compat.v1.GraphDef()
  print ("graph is in: ",params["FROZEN_GRAPH"]) 
  with tf.compat.v1.gfile.GFile(params["FROZEN_GRAPH"], 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.compat.v1.import_graph_def(graph_def, name='')


def get_handles_to_tensors():
  graph = tf.compat.v1.get_default_graph()
  ops = graph.get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  key_list = [
    'num_detections',
    'detection_boxes',
    'detection_scores',
    'detection_classes'
  ]
  for key in key_list:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
  image_tensor = graph.get_tensor_by_name('image_tensor:0')
  return tensor_dict, image_tensor


def load_image(image_files,iter_num,processed_image_ids,params):

  image_file = image_files[iter_num]
  image_id = ck_utils.filename_to_id(image_file, params["DATASET_TYPE"])
  processed_image_ids.append(image_id)
  image_path = os.path.join(params["IMAGES_DIR"], image_file)
  img = cv2.imread(image_path)
  orig_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # Load to numpy array, separately from original image
  image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
  # Make batch from single image
  im_height, im_width, _ = image.shape
  batch_shape = (1, im_height, im_width, 3)
  batch_data = image.reshape(batch_shape) 
  return batch_data,processed_image_ids,(im_width, im_height),orig_image


def save_detection_txt(image_file, image_size, output_dict, category_index, params):
  (im_width, im_height) = image_size
  file_name = os.path.splitext(image_file)[0]
  res_file = os.path.join(params["DETECTIONS_OUT_DIR"], file_name) + '.txt'
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
      f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:d} {}\n'\
        .format(x1*im_width, y1*im_height, x2*im_width, y2*im_height, score, class_id, class_name))


def save_detection_img(image_file, image_np, output_dict, category_index,params):
  if not params["SAVE_IMAGES"]: return

  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      line_thickness=2)
  cv2.imwrite(os.path.join(params["IMAGES_OUT_DIR"],image_file),cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))



def postprocess_image(image_files, iter_num, image_size,dummy, image_data,output_dict, category_index,params):
  # All outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  image_file = image_files[iter_num]
  save_detection_txt(image_file, image_size, output_dict, category_index,params)
  save_detection_img(image_file, image_data[0], output_dict, category_index,params)


###### functions for batch processing

def load_images_batch(image_list,iter_num,processed_image_ids,params):
  batch_data = []
  batch_sizes = []
  for img in range(params["BATCH_SIZE"]):
    img_rd = cv2.imread(os.path.join(params["IMAGES_DIR"], image_list[iter_num*params["BATCH_SIZE"]+img]))
    
    image = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB).astype(np.uint8)
    im_height, im_width, _ = image.shape
    batch_sizes.append((im_width, im_height))
    image = cv2.resize(image,(params["RESIZE_WIDTH_SIZE"],params["RESIZE_HEIGHT_SIZE"]))
    image_id = ck_utils.filename_to_id(image_list[iter_num*params["BATCH_SIZE"]+img], params["DATASET_TYPE"])
    processed_image_ids.append(image_id)
    img_data = image.reshape((params["RESIZE_HEIGHT_SIZE"],params["RESIZE_WIDTH_SIZE"],3))
    batch_data.append(img_data)
  return batch_data,processed_image_ids,batch_sizes,image_id #last value is not needed actually.

#TODO make the save_detection_img able to resize to the original dimensions
def postprocess_batch(image_files, iter_num, image_size,dummy, image_data,output_dict, category_index,params):

  for img in range(params["BATCH_SIZE"]):
    tmp_output_dict={}
    tmp_output_dict['num_detections'] = int(output_dict['num_detections'][img])
    tmp_output_dict['detection_classes'] = output_dict['detection_classes'][img].astype(np.uint8)
    tmp_output_dict['detection_boxes'] = output_dict['detection_boxes'][img]
    tmp_output_dict['detection_scores'] = output_dict['detection_scores'][img]
    
    save_detection_txt(image_files[iter_num*params["BATCH_SIZE"]+img], image_size[img], tmp_output_dict, category_index,params)
    save_detection_img(image_files[iter_num*params["BATCH_SIZE"]+img], image_data[img], tmp_output_dict, category_index,params)
 




def detect(category_index, func_defs):
  # Prepare TF config options
  tf_config = make_tf_config()

  # Prepare directories
  ck_utils.prepare_dir(params["RESULTS_OUT_DIR"])
  ck_utils.prepare_dir(params["ANNOTATIONS_OUT_DIR"])
  ck_utils.prepare_dir(params["IMAGES_OUT_DIR"])
  ck_utils.prepare_dir(params["DETECTIONS_OUT_DIR"])

  # Load processing image filenames

  image_files = ck_utils.load_image_list(params["IMAGES_DIR"], params["BATCH_COUNT"]*params["BATCH_SIZE"], params["SKIP_IMAGES"])

  with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:
    setup_time_begin = time.time()
    
    # Make TF graph def from frozen graph file
    begin_time = time.time()
    # FIRST HOOK: load graph
    func_defs["load_graph"](params)
    graph_load_time = time.time() - begin_time
    print('Graph loaded in {:.4f}s'.format(graph_load_time))
#    print('custom model is: ',CUSTOM_MODEL)
    #SECOND HOOK: get tensors
    tensor_dict, input_tensor = func_defs["get_tensor"]()
    setup_time = time.time() - setup_time_begin
    print ("setup time is",setup_time)
    ###### END SETUP PHASE
    # Process images
    test_time_begin = time.time()
    image_index = 0
    load_time_total = 0
    detect_time_total = 0
    images_processed = 0
    processed_image_ids = []
    loop_limit = len(image_files) if (params["ENABLE_BATCH"]) == 0 else params["BATCH_COUNT"]  #defines loop boundary, that is different if batch or non batch processing are involved. the structure of the loop is however the same.
    for iter_num in range (loop_limit):
 
      load_time_begin = time.time()
      # THIRD HOOK: preprocess
      image_data,processed_image_ids,image_size,original_image = func_defs["preprocess"](image_files,iter_num,processed_image_ids,params)
      
      load_time = time.time() - load_time_begin
      load_time_total += load_time
      # Detect image: common
      detect_time_begin = time.time()
      feed_dict = {input_tensor: image_data}
      output_dict = sess.run(tensor_dict, feed_dict)
      #FOURTH HOOK: convert from tensorRT to normal dict
      output_dict =func_defs["out_conv"](output_dict)
      
      detect_time = time.time() - detect_time_begin
      # Exclude first image from averaging
      if iter_num > 0 or params["BATCH_COUNT"] == 1:
        detect_time_total += detect_time
        images_processed += 1## may be revision needed
 
      # FIFTH hook: process results
      func_defs["postprocess"](image_files,iter_num, image_size,original_image,image_data,output_dict, category_index, params)

      if params["FULL_REPORT"]:
        print('Detected in {:.4f}s'.format(detect_time))

  # Save processed images ids list to be able to run
  # evaluation without repeating detections (CK_SKIP_DETECTION=YES)
  with open(params["IMAGE_LIST_FILE"], 'w') as f:
    f.write(json.dumps(processed_image_ids))

  test_time = time.time() - test_time_begin
  detect_avg_time = detect_time_total / images_processed
  load_avg_time = load_time_total / len(processed_image_ids)
  OPENME = {}
  OPENME['setup_time_s'] = setup_time
  OPENME['test_time_s'] = test_time
  OPENME['graph_load_time_s'] = graph_load_time
  OPENME['images_load_time_total_s'] = load_time_total
  OPENME['images_load_time_avg_s'] = load_avg_time
  OPENME['detection_time_total_s'] = detect_time_total
  OPENME['detection_time_avg_s'] = detect_avg_time
  OPENME['avg_time_ms'] = detect_avg_time * 1000
  OPENME['avg_fps'] = 1.0 / detect_avg_time if detect_avg_time > 0 else 0

  with open(params["TIMER_JSON"], 'w') as o:
    json.dump(OPENME, o, indent=2, sort_keys=True)

  return processed_image_ids





def no_conv(output_dict):
  return output_dict


'''#Init function. the dictionary approach doesn't work since it wants all the function defined. 
inside the function, all the hooks are actually assigned to the right functions. some functions can be shared between configurations.
This model supports 8 possible configuration, up to now:
 0: classical, with no batch and with model coming from tensorflow zoo
 1: batched, with model from tensorflow zoo
 2: custom model, without batch
 3: custom model, with batch
 4: classical model, no batch, tensorRT
 5: classical model, batch, tensorRT
 6: custom model, no batch, tensorRT
 7: custom model, batch, tensorRT
IMPORTANT: for custom model, is up to the developer to provide implementations of the hook functions, that must be divided into 2 python files.
the first files must contain the preprocess, postprocess and get_handles_to_tensor functions, and will work without tensorRT
the second file must provide the tensorRT support, so the get_handles_to_tensor_RT, the convert_from_tensorrt and the load_grap_tensorrt_custom functions must be provided.

Function descriptions and parameters:

-preprocess:
in charge or preparing the image for the detection. must produce the input tensor and some other helper data.
	in:
		image_files          -> list with all the filenames of the image to process, with full path
		iter_num             -> integer with the loop iteration value
		processed_image_ids  -> list with the ids of all the processed images, it's an in-out parameter (the function must append to this)
		params               -> dictionary with the application parameters
	out:
		image_data           -> numpy array to be fed to the detection graph (input tensor)
		processed_image_ids  -> see input parameters
		image_size           -> [list of] tuple with the sizes. depends if batch is used or not, if not is a single tuple
		original_image       -> [list of] list containing the original images as read before the modification done in preprocessing. may be useless

-postprocess:
in charge of producing the output of the detection. must read output tensors and produce the txt file with the detections, and if required the images with the boxes.
	in:
		image_files	     -> list with all the filenames of the image to process, with full path
		iter_num             -> integer with the loop iteration value
		image_size	     -> [list of] tuple with the sizes. depends if batch is used or not, if not is a single tuple
		original_image       -> [list of] list containing the original images as read before the modification done in preprocessing. may be useless
		image_data           -> numpy array to be fed to the detection graph (input tensor)
		output_dict          -> output tensors. dictionary containing the tensors as "name : value" couples.
		category_index       -> dictionary to identify label and categories
		params               -> dictionary with the application parameters
	out:
		------

-get_tensors
in charge of getting the input and output tensors from the model graph.
	in: 
		------
	out:
		tensor_dict          -> dictionary with the output tensors
		input_tensor         -> input tensor


-out_conv
in charge of converting the dictionary if tensorRT is used, since output in tensorRT is a list and not a dict
	in:
		output_dict          -> output tensors. if tensorRT, is a list containing the output tensors 

	out:
		output_dict          -> output tensors. dictionary containing the tensors as "name : value" couples.


-load_graph:
in charge of loading the graph from a frozen model.
	in:
		params               -> dictionary with the application parameters
	
	out:
		------

'''
def init(params):
  func_defs = {}
  if params["ENABLE_BATCH"] == 0:
    ## non batch mode
    if params["CUSTOM_MODEL"]==0:
      ##non custom
      if params["WITH_TENSORRT"] == 0:
         #non tensorRT
         print ("inside init")
         func_defs["postprocess"] = postprocess_image
         func_defs["preprocess"]  = load_image
         func_defs["get_tensor"]  = get_handles_to_tensors
         func_defs["load_graph"]  = load_graph_traditional
         func_defs["out_conv"]    = no_conv
      else:
         import tensorRT_hooks
         func_defs["postprocess"] = postprocess_image
         func_defs["preprocess"]  = load_image
         func_defs["get_tensor"]  = tensorRT_hooks.get_handles_to_tensors_RT
         func_defs["load_graph"]  = tensorRT_hooks.load_graph_tensorrt
         func_defs["out_conv"]    = tensorRT_hooks.convert_from_tensorrt
    else:
      ##custom
      import custom_hooks
      if params["WITH_TENSORRT"] == 0:
         #non tensorRT
         func_defs["postprocess"] = custom_hooks.ck_custom_postprocess
         func_defs["preprocess"]  = custom_hooks.ck_custom_preprocess
         func_defs["get_tensor"]  = custom_hooks.ck_custom_get_tensors
         func_defs["load_graph"]  = load_graph_traditional
         func_defs["out_conv"]    = no_conv
      else:
         import custom_tensorRT
         func_defs["postprocess"] = custom_hooks.ck_custom_postprocess
         func_defs["preprocess"]  = custom_hooks.ck_custom_preprocess
         func_defs["get_tensor"]  = custom_tensorRT.get_handles_to_tensors_RT
         func_defs["load_graph"]  = custom_tensorRT.load_graph_tensorrt_custom
         func_defs["out_conv"]    = custom_tensorRT.convert_from_tensorrt

  else:
    ##  batch mode
    if params["CUSTOM_MODEL"]==0:
      ##non custom
      if params["WITH_TENSORRT"] == 0:
         #non tensorRT
         func_defs["postprocess"] = postprocess_batch
         func_defs["preprocess"]  = load_images_batch
         func_defs["get_tensor"]  = get_handles_to_tensors
         func_defs["load_graph"]  = load_graph_traditional
         func_defs["out_conv"]    = no_conv
      else:
         import tensorRT_hooks
         func_defs["postprocess"] = postprocess_batch
         func_defs["preprocess"]  = load_images_batch
         func_defs["get_tensor"]  = tensorRT_hooks.get_handles_to_tensors_RT
         func_defs["load_graph"]  = tensorRT_hooks.load_graph_tensorrt
         func_defs["out_conv"]    = tensorRT_hooks.convert_from_tensorrt
    else:
      ##custom
      import custom_hooks
      if params["WITH_TENSORRT"] == 0:
         #non tensorRT
         func_defs["postprocess"] = custom_hooks.ck_custom_postprocess_batch
         func_defs["preprocess"]  = custom_hooks.ck_custom_preprocess_batch
         func_defs["get_tensor"]  = custom_hooks.ck_custom_get_tensors
         func_defs["load_graph"]  = load_graph_traditional
         func_defs["out_conv"]    = no_conv               
      else:
         import custom_tensorRT
         func_defs["postprocess"] = custom_hooks.ck_custom_postprocess_batch
         func_defs["preprocess"]  = custom_hooks.ck_custom_preprocess_batch
         func_defs["get_tensor"]  = custom_tensorRT.get_handles_to_tensors_RT
         func_defs["load_graph"]  = custom_tensorRT.load_graph_tensorrt_custom
         func_defs["out_conv"]    = custom_tensorRT.convert_from_tensorrt
  return func_defs


def main(_):
  # Print settings
  print("Model frozen graph: " + params["FROZEN_GRAPH"])
  print("Model label map file: " + params["LABELMAP_FILE"])
  print("Model is for dataset: " + params["MODEL_DATASET_TYPE"])

  print("Dataset images: " + params["IMAGES_DIR"])
  print("Dataset annotations: " + params["ANNOTATIONS_PATH"])
  print("Dataset type: " + params["DATASET_TYPE"])

  print('Image count: {}'.format(params["BATCH_COUNT"]*params["BATCH_SIZE"]))
  print("Metric type: " + params["METRIC_TYPE"])
  print('Results directory: {}'.format(params["RESULTS_OUT_DIR"]))
  print("Temporary annotations directory: " + params["ANNOTATIONS_OUT_DIR"])
  print("Detections directory: " + params["DETECTIONS_OUT_DIR"])
  print("Result images directory: " + params["IMAGES_OUT_DIR"])
  print('Save result images: {}'.format(params["SAVE_IMAGES"]))

  # Create category index
  category_index = label_map_util.create_category_index_from_labelmap(params["LABELMAP_FILE"], use_display_name=True)
  categories_list = category_index.values() # array: [{"id": 88, "name": "teddy bear"}, ...]
  print('Categories: {}'.format(categories_list))

 # Init function: set the correct imports and function pointers.
  func_defs = init(params)

  # Run detection if needed
  ck_utils.print_header('Process images')
  if params["SKIP_DETECTION"]:
    print('\nSkip detection, evaluate previous results')
  else:
    processed_image_ids = detect(category_index,func_defs)
  
  ENV={}
  ENV['PYTHONPATH'] = os.getenv('PYTHONPATH')
  ENV['LABELMAP_FILE'] = params["LABELMAP_FILE"]
  ENV['MODEL_DATASET_TYPE'] = params["MODEL_DATASET_TYPE"]
  ENV['DATASET_TYPE'] = params["DATASET_TYPE"]
  ENV['ANNOTATIONS_PATH'] = params["ANNOTATIONS_PATH"]
  ENV['METRIC_TYPE'] = params["METRIC_TYPE"]
  ENV['IMAGES_OUT_DIR'] = params["IMAGES_OUT_DIR"]
  ENV['DETECTIONS_OUT_DIR'] = params["DETECTIONS_OUT_DIR"]
  ENV['ANNOTATIONS_OUT_DIR'] = params["ANNOTATIONS_OUT_DIR"]
  ENV['RESULTS_OUT_DIR'] = params["RESULTS_OUT_DIR"]
  ENV['FULL_REPORT'] = params["FULL_REPORT"]
  ENV['IMAGE_LIST_FILE'] = params["IMAGE_LIST_FILE"]
  ENV['TIMER_JSON'] = params["TIMER_JSON"]

  with open(params["ENV_JSON"], 'w') as o:
    json.dump(ENV, o, indent=2, sort_keys=True)

if __name__ == '__main__':
  tf.compat.v1.app.run()
