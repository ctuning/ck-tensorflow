#
# CK configuration script for TensorFlow package
#
# Developer(s): 
#  * Grigori Fursin, dividiti/cTuning foundation
#

import os
import sys
import json

##############################################################################
# customize installation

def setup(i):
    """
    Input:  {
              cfg              - meta of this soft entry
              self_cfg         - meta of module soft
              ck_kernel        - import CK kernel module (to reuse functions)

              host_os_uoa      - host OS UOA
              host_os_uid      - host OS UID
              host_os_dict     - host OS meta

              target_os_uoa    - target OS UOA
              target_os_uid    - target OS UID
              target_os_dict   - target OS meta

              target_device_id - target device ID (if via ADB)

              tags             - list of tags used to search this entry

              env              - updated environment vars from meta
              customize        - updated customize vars from meta

              deps             - resolved dependencies for this soft

              interactive      - if 'yes', can ask questions, otherwise quiet

              path             - path to entry (with scripts)
              install_path     - installation path
            }

    Output: {
              return        - return code =  0, if successful
                                          >  0, if error
              (error)       - error text if return > 0
              (install-env) - prepare environment to be used before the install script
            }

    """

    # Get variables
    ck=i['ck_kernel']

    # Find entry with script to reuse
    r=ck.access({'action':'find', 'cid':'package:4b44e797606a9488'})
    if r['return']>0: return r

    ppp=r['path']

    r=ck.load_module_from_path({'path':ppp, 'module_code_name':'custom', 'skip_init':'yes'})
    if r['return']>0: return r

    script=r['code']

    return script.setup(i)
