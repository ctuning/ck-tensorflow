#
# Save output of image_benchmark.py to the CK timing format.
#
# Developer(s):
#   - Anton Lokhmotov, dividiti, 2017
#

import json
import os
import re
import sys

def ck_postprocess(i):
    ck=i['ck_kernel']

    # Dictionary to return.
    d={}

    # Load output from file.
    rt=i['run_time']
    r=ck.load_json_file({'json_file':rt['fine_grain_timer_file']})
    if r['return']>0: return r
    d=r['dict']

    # Save environment variables.
    env=i.get('env',{})
    d['env']={ var : env[var] for var in env }

    # Save metrics of interest.
    d['avg_fps']=1/d['avg_time_s']
    d['avg_time_ms']=d['avg_time_s']*1e3
    d['total_time_ms']=d['avg_time_ms']*d['env']['CK_BATCH_SIZE']

    # Save internal CK keys.
    d['execution_time']=d['avg_time_s']*d['env']['CK_BATCH_SIZE']
    d['post_processed']='yes'

    # Save augmented output back to file.
    rr={}
    rr['return']=0
    if d.get('post_processed','')=='yes':
        r=ck.save_json_to_file({'json_file':'tmp-ck-timer.json', 'dict':d})
        if r['return']>0: return r
    else:
        rr['return']=1
        rr['error']='failed to postprocess \`image_benchmark\' command'

    return rr

# Do not add anything here!
