#
# Copyright (c) 2017 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import imp
import os
import tensorflow as tf

MODEL_MODULE = os.getenv('CK_ENV_TENSORFLOW_MODEL_MODULE')
MODEL_WEIGHTS = os.getenv('CK_ENV_TENSORFLOW_MODEL_WEIGHTS')
TARGET_FILE = os.getenv('CK_TARGET_PB_FILE')
FREEZE_AS_TEXT = os.getenv('CK_FREEZE_AS_TEXT')

from tensorflow.python.platform import gfile

if FREEZE_AS_TEXT == "YES":
  from google.protobuf import text_format

def main(_):
  print('Model module: ' + MODEL_MODULE)
  print('Model weights: ' + MODEL_WEIGHTS)

  target_file = TARGET_FILE if TARGET_FILE else 'graph.pb'
  print('Target file: ' + target_file)

  # Load model implementation module
  model = imp.load_source('tf_model', MODEL_MODULE)

  # Load weights
  model.load_weights(MODEL_WEIGHTS)

  with tf.Graph().as_default(), tf.Session() as sess:
    # Build net
    input_shape = (None, 227, 227, 3)
    input_node = tf.placeholder(dtype=tf.float32, shape=input_shape, name="input")
    output_node = model.inference(input_node)

    graph_def = tf.get_default_graph().as_graph_def()

    if FREEZE_AS_TEXT == "YES":
      print('Serialize as text...')
      with gfile.GFile(target_file, "w") as f:
        f.write(text_format.MessageToString(graph_def))
    else:
      print('Serialize as binary...')
      with gfile.GFile(target_file, "wb") as f:
        f.write(graph_def.SerializeToString())
    
    print('OK')

if __name__ == '__main__':
  tf.app.run()
