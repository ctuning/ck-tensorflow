#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os

def ck_preprocess(i):
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'].get(var)
  
  MODEL_TFLITE_FILE = ''
  # Our tensorflow model packages provide model as checkpoints files.
  # But we have to find tflite graph file in model's directory.
  # If weights will be already provided as tflite file,
  # the code will still be working even though a bit excessive.
  MODEL_DIR = dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_ROOT')
  for filename in os.listdir(MODEL_DIR):
    if filename.endswith('.tflite'):
      MODEL_TFLITE_FILE = filename
      MODEL_TFLITE_PATH = os.path.join(MODEL_DIR, MODEL_TFLITE_FILE)
  if not MODEL_TFLITE_FILE:
    return {'return': 1, 'error': 'Tflite graph is not found in the selected model package'}

  # Setup parameters for program
  new_env = {}
  files_to_push_by_path = {}

  if i['target_os_dict'].get('remote','') == 'yes':
    # For Android we need only filename without full path  
    new_env['RUN_OPT_GRAPH_FILE'] = MODEL_TFLITE_FILE
    files_to_push_by_path['RUN_OPT_GRAPH_PATH'] = MODEL_TFLITE_PATH
  else:
    new_env['RUN_OPT_GRAPH_FILE'] = MODEL_TFLITE_PATH

  print('--------------------------------\n')
  return {
    'return': 0,
    'new_env': new_env,
    'files_to_push_by_path': files_to_push_by_path,
  }
