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

  LABELS_FILE = 'labels.txt'

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
    return {'return': 1, 'error': 'Tflite graph is not found in selected model package'}

  new_env = {}
  files_to_push = []

  if i['target_os_dict'].get('remote','') == 'yes':
    # For Android we need only filename without full path  
    new_env['CK_ENV_TENSORFLOW_MODEL_TFLITE'] = MODEL_TFLITE_FILE
    new_env['CK_ENV_LABELS_FILE'] = LABELS_FILE

    # Set list of additional files to be copied to Android device.
    # We have to set these files via env variables with full paths 
    # in order to they will be copied into remote program dir without sub-paths.
    new_env['CK_ENV_TENSORFLOW_MODEL_TFLITE_PATH'] = MODEL_TFLITE_PATH
    new_env['CK_ENV_LABELS_FILE_PATH'] = os.path.join(os.getcwd(), '..', LABELS_FILE)
    files_to_push.append("$<<CK_ENV_TENSORFLOW_MODEL_TFLITE_PATH>>$")
    files_to_push.append("$<<CK_ENV_LABELS_FILE_PATH>>$")
  else:
    new_env['CK_ENV_TENSORFLOW_MODEL_TFLITE'] = MODEL_TFLITE_PATH
    new_env['CK_ENV_LABELS_FILE'] = os.path.join('..', LABELS_FILE)

  return {'return': 0, 'new_env': new_env, 'run_input_files': files_to_push}

