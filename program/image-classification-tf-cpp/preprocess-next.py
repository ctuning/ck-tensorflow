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

  # Our tensorflow model packages provide model as checkpoints files.
  # But we have to find frozen graph file in model's directory.
  # If weights will be already provided as frozen file,
  # the code will still be working even though a bit excessive.
  MODEL_DIR = dep_env('weights', 'CK_ENV_TENSORFLOW_MODEL_ROOT')
  for filename in os.listdir(MODEL_DIR):
    if filename.endswith('.pb'):
      MODEL_FROZEN_FILE = filename
      MODEL_FROZEN_PATH = os.path.join(MODEL_DIR, filename)
    elif filename.endswith('_info.txt'):
      # Read input and output layer names from graph info file
      with open(os.path.join(MODEL_DIR, filename), 'r') as f:
        for line in f:
          key_name = line.split(' ')
          if len(key_name) == 2:
            if key_name[0] == 'Input:':
              INPUT_LAYER_NAME = key_name[1].strip()
            elif key_name[0] == 'Output:':
              OUTPUT_LAYER_NAME = key_name[1].strip()
                                                       
  
  if not MODEL_FROZEN_FILE:
    return {'return': 1, 'error': 'Frozen graph is not found in the selected model package'}
  if not INPUT_LAYER_NAME:
    return {'return': 1, 'error': 'Input layer name is not set, check `*_info.txt` file in the selected model package'}
  if not OUTPUT_LAYER_NAME:
    return {'return': 1, 'error': 'Output layer name is not set, check `*_info.txt` file in the selected model package'}

  # Setup parameters for program
  new_env = {}
  files_to_push_by_path = {}

  if i['target_os_dict'].get('remote','') == 'yes':
    # For Android we need only filename without full path  
    new_env['RUN_OPT_GRAPH_FILE'] = MODEL_FROZEN_FILE
    files_to_push_by_path['RUN_OPT_GRAPH_PATH'] = MODEL_FROZEN_PATH
  else:
    new_env['RUN_OPT_GRAPH_FILE'] = MODEL_FROZEN_PATH

  new_env['RUN_OPT_INPUT_LAYER_NAME'] = INPUT_LAYER_NAME
  new_env['RUN_OPT_OUTPUT_LAYER_NAME'] = OUTPUT_LAYER_NAME

  print('--------------------------------\n')
  return {
    'return': 0,
    'new_env': new_env,
    'files_to_push_by_path': files_to_push_by_path,
  }
