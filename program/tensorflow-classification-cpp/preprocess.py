#
# Preprocessing Caffe templates
#
# Developer: Grigori Fursin, cTuning foundation, 2016
#

import json
import os
import re

def ck_preprocess(i):

    ck=i['ck_kernel']
    rt=i['run_time']
    deps=i['deps']

    env=i['env']
    nenv={} # new environment to be added to the run script

    hosd=i['host_os_dict']
    tosd=i['target_os_dict']
    remote=tosd.get('remote','')

    if remote=='yes':
       es=tosd['env_set']
    else:
       es=hosd['env_set'] # set or export

    b='\n'

    # Find template
    x=deps['tensorflow-aux']

    pb_path_full=x['dict']['env']['CK_ENV_DATASET_TENSORFLOW_AUX'] + '/tensorflow_inception_graph.pb'
    nenv['CK_ENV_DATASET_TENSORFLOW_AUX_PB']=pb_path_full

    if remote=='yes':
       # For Android we need only filename without full path
       pb_path=os.path.basename(pb_path_full)
    else:
       pb_path=pb_path_full

    nenv['CK_DATASET_TENSORFLOW_AUX_PB']=pb_path

    try:
        cur_dir=os.getcwd()
    except OSError:
        os.chdir('..')
        cur_dir=os.getcwd()

    return {'return':0, 'new_env':nenv}

# Do not add anything here!
