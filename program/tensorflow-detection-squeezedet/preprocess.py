#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#  

import os
import json

########################################################################
# Search and select a KITTI evaluation program

def get_kitti_eval_program(ck):
  res = ck.access({'action': 'search',
                   'module_uoa': 'program',
                   'tags': 'kitti-eval-tool'})
  if res['return'] > 0:
    return res

  selected_prog = None

  progs = res['lst']
  if len(progs) > 0:
    if len(progs) == 1:
      selected_prog = progs[0]
    else:
      ck.out('\nMore than one program is found:\n')
      res = ck.access({'action': 'select_uoa',
                       'module_uoa': 'choice',
                       'choices': progs})
      if res['return'] > 0:
        return res

      for d in progs:
        if d['data_uid'] == res['choice']:
          selected_prog = d
          break

  return selected_prog


########################################################################
# Load meta of some ck-entry

def load_meta(entry):
  with open(os.path.join(entry['path'], '.cm', 'meta.json')) as f:
    return json.load(f)


########################################################################

def ck_preprocess(i):
  ck = i['ck_kernel']

  prog = get_kitti_eval_program(ck)
  if not prog:
    ck.out('\nNo KITTI evaluation tool is found, only default evaluation is available.\n')
    return {'return': 0}

  prog_path = prog['path']
  ck.out('Checking KITTI evaluation tool in {} ...'.format(prog_path))
    
  prog_path = os.path.join(prog_path, 'tmp')
  if not os.path.isdir(prog_path):
    ck.out('Seems program is not compiled, only default evaluation is available.\n')
    return {'return': 0}
  
  meta = load_meta(prog)
  prog_file = meta['target_file']
  prog_path = os.path.join(prog_path, prog_file)
  if not os.path.isfile(prog_path):
    ck.out('Seems program is not compiled, only default evaluation is available.\n')
    return {'return': 0}

  ck.out('Executable found: {}'.format(prog_path))
  os.environ['CK_KITTI_EVAL_TOOL'] = os.path.join(prog_path)

  return {'return': 0}
