#!/usr/bin/env python

import sys
import tensorflow as tf

input_tf_model_filepath = sys.argv[1]
output_tflite_model_filepath = sys.argv[2]

# FIXME: keep them as magic constsants
#
input_arrays = ["input_tensor"]
output_arrays = ["softmax_tensor", "ArgMax"]

converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
  input_tf_model_filepath, input_arrays, output_arrays)
tflite_model = converter.convert()
open(output_tflite_model_filepath, "wb").write(tflite_model)
