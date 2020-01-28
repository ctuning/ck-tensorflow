#!/usr/bin/env python3

import json
import time
import os
import shutil
import numpy as np
import tensorflow as tf


## Model properties:
#
MODEL_PATH              = os.environ['CK_ENV_TENSORFLOW_MODEL_TF_FROZEN_FILEPATH']
INPUT_LAYER_NAME        = os.environ['CK_ENV_TENSORFLOW_MODEL_INPUT_LAYER_NAME']
OUTPUT_LAYER_NAME       = os.environ['CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME']
MODEL_DATA_LAYOUT       = os.environ['ML_MODEL_DATA_LAYOUT']
IMAGE_SIZE              = int(os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_INPUT_SQUARE_SIDE'))
LABELS_PATH             = os.environ['CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT']

## Image normalization:
#
MODEL_NORMALIZE_DATA    = os.getenv("CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA") in ('YES', 'yes', 'ON', 'on', '1')
SUBTRACT_MEAN           = os.getenv("CK_ENV_TENSORFLOW_MODEL_SUBTRACT_MEAN") in ('YES', 'yes', 'ON', 'on', '1')
GIVEN_CHANNEL_MEANS     = os.getenv("ML_MODEL_GIVEN_CHANNEL_MEANS", '')
if GIVEN_CHANNEL_MEANS:
    GIVEN_CHANNEL_MEANS = np.array(GIVEN_CHANNEL_MEANS.split(' '), dtype=np.float32)

## Input image properties:
#
IMAGE_DIR               = os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR')
IMAGE_LIST              = os.path.join(IMAGE_DIR, os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF'))
IMAGE_DATA_TYPE         = np.dtype( os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_DATA_TYPE', 'uint8') )

## Writing the results out:
#
RESULT_DIR              = os.getenv('CK_RESULTS_DIR')
FULL_REPORT             = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')

## Processing in batches:
#
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))
BATCH_SIZE              = int(os.getenv('CK_BATCH_SIZE', 1))
BATCHED_VOLUME          = BATCH_COUNT * BATCH_SIZE
SUBSET_VOLUME           = int(os.getenv('CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_VOLUME') or '0') or BATCHED_VOLUME


if BATCHED_VOLUME > SUBSET_VOLUME:
    print('*'*30)
    print('Image set size is: {}'.format(SUBSET_VOLUME))
    print('BATCH_COUNT * BATCH_SIZE is: {}'.format(BATCHED_VOLUME))
    BATCH_COUNT = int (SUBSET_VOLUME / BATCH_SIZE)
    print('BATCH_COUNT restricted to: {}'.format(BATCH_COUNT))
    print('*'*30)

def load_preprocessed_batch(image_list, image_index):
    batch_data = []
    for _ in range(BATCH_SIZE):
        img_file = os.path.join(IMAGE_DIR, image_list[image_index])
        img = np.fromfile(img_file, IMAGE_DATA_TYPE)
        img = img.reshape((IMAGE_SIZE, IMAGE_SIZE, 3))

        if IMAGE_DATA_TYPE != 'float32':
            img = img.astype(np.float32)

            # Normalize
            if MODEL_NORMALIZE_DATA:
                img = img/127.5 - 1.0

            # Subtract mean value
            if SUBTRACT_MEAN:
                if len(GIVEN_CHANNEL_MEANS):
                    img -= GIVEN_CHANNEL_MEANS
                else:
                    img -= np.mean(img)

        # Add img to batch
        batch_data.append( [img] )
        image_index += 1

    nhwc_data = np.concatenate(batch_data, axis=0)

    if MODEL_DATA_LAYOUT == 'NHWC':
        #print(nhwc_data.shape)
        return nhwc_data, image_index
    else:
        nchw_data = nhwc_data.transpose(0,3,1,2)
        #print(nchw_data.shape)
        return nchw_data, image_index


def load_graph(frozen_graph_filename):

    with tf.compat.v1.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # import the graph_def into a new Graph and return it
    with tf.Graph().as_default() as graph:
        # The value of name variable will prefix every op/node name. The default is "import".
        # Since we don't want any prefix, we have to override it with an empty string.
        tf.import_graph_def(graph_def, name="")

    return graph


def load_labels(labels_filepath):
    my_labels = []
    input_file = open(labels_filepath, 'r')
    for l in input_file:
        my_labels.append(l.strip())
    return my_labels


def main():
    print('Images dir: ' + IMAGE_DIR)
    print('Image list: ' + IMAGE_LIST)
    print('Image size: {}'.format(IMAGE_SIZE))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Batch count: {}'.format(BATCH_COUNT))
    print('Result dir: ' + RESULT_DIR);
    print('Normalize: {}'.format(MODEL_NORMALIZE_DATA))
    print('Subtract mean: {}'.format(SUBTRACT_MEAN))
    print('Per-channel means to subtract: {}'.format(GIVEN_CHANNEL_MEANS))

    labels = load_labels(LABELS_PATH)
    num_labels = len(labels)

    # Prepare TF config options
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = float(os.getenv('CK_TF_GPU_MEMORY_PERCENT', 33)) / 100.0
    num_processors = int(os.getenv('CK_TF_CPU_NUM_OF_PROCESSORS', 0))
    if num_processors > 0:
        config.device_count["CPU"] = num_processors

    # Load preprocessed image filenames:
    with open(IMAGE_LIST, 'r') as f:
        image_list = [ s.strip() for s in f ]

    # Cleanup results directory
    if os.path.isdir(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.mkdir(RESULT_DIR)

    setup_time_begin = time.time()

    # Load the TF model from ProtoBuf file
    graph = load_graph(MODEL_PATH)
    input_layer = graph.get_tensor_by_name(INPUT_LAYER_NAME+':0')
    output_layer = graph.get_tensor_by_name(OUTPUT_LAYER_NAME+':0')

    model_input_shape = input_layer.shape
    model_output_shape = output_layer.shape
    model_classes = model_output_shape[1]
    bg_class_offset = model_classes-len(labels)  # 1 means the labels represent classes 1..1000 and the background class 0 has to be skipped

    if MODEL_DATA_LAYOUT == 'NHWC':
        (samples, height, width, channels) = model_input_shape
    else:
        (samples, channels, height, width) = model_input_shape

    print("Data layout: {}".format(MODEL_DATA_LAYOUT) )
    print("Input layer: {}".format(input_layer) )
    print("Output layer: {}".format(output_layer) )
    print("Expected input shape: {}".format(model_input_shape) )
    print("Output layer shape: {}".format(model_output_shape) )
    print("Number of labels: {}".format(num_labels))
    print("Background/unlabelled classes to skip: {}".format(bg_class_offset))
    print("Data normalization: {}".format(MODEL_NORMALIZE_DATA) )
    print("")

    with tf.compat.v1.Session(graph=graph, config=config) as sess:

        setup_time = time.time() - setup_time_begin

        # Run batched mode
        test_time_begin = time.time()
        image_index = 0
        load_total_time = 0
        classification_total_time = 0
        images_loaded = 0
        images_processed = 0
        for batch_index in range(BATCH_COUNT):
            batch_number = batch_index+1
            if FULL_REPORT or (batch_number % 10 == 0):
                print("\nBatch {} of {}".format(batch_number, BATCH_COUNT))
          
            begin_time = time.time()
            batch_data, image_index = load_preprocessed_batch(image_list, image_index)
            load_time = time.time() - begin_time
            load_total_time += load_time
            images_loaded += BATCH_SIZE
            if FULL_REPORT:
                print("Batch loaded in %fs" % (load_time))

            # Classify batch
            begin_time = time.time()
            batch_results = sess.run(output_layer, feed_dict={ input_layer: batch_data } )
            classification_time = time.time() - begin_time
            if FULL_REPORT:
                print("Batch classified in %fs" % (classification_time))
          
            # Exclude first batch from averaging
            if batch_index > 0 or BATCH_COUNT == 1:
                classification_total_time += classification_time
                images_processed += BATCH_SIZE

            # Process results
            for index_in_batch in range(BATCH_SIZE):
                softmax_vector = batch_results[index_in_batch][bg_class_offset:]    # skipping the background class on the left (if present)
                global_index = batch_index * BATCH_SIZE + index_in_batch
                res_file = os.path.join(RESULT_DIR, image_list[global_index])
                with open(res_file + '.txt', 'w') as f:
                    for prob in softmax_vector:
                        f.write('{}\n'.format(prob))
            
    test_time = time.time() - test_time_begin
    classification_avg_time = classification_total_time / images_processed
    load_avg_time = load_total_time / images_loaded


    # Store benchmarking results:
    output_dict = {
        'setup_time_s': setup_time,
        'test_time_s': test_time,
        'images_load_time_total_s': load_total_time,
        'images_load_time_avg_s': load_avg_time,
        'prediction_time_total_s': classification_total_time,
        'prediction_time_avg_s': classification_avg_time,

        'avg_time_ms': classification_avg_time * 1000,
        'avg_fps': 1.0 / classification_avg_time,
        'batch_time_ms': classification_avg_time * 1000 * BATCH_SIZE,
        'batch_size': BATCH_SIZE,
    }
    with open('tmp-ck-timer.json', 'w') as out_file:
        json.dump(output_dict, out_file, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
