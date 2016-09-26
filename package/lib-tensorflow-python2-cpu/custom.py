#!/usr/bin/python

#
# Developer: Zaborovskiy Vladislav, vladzab@yandex.ru
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
    s=''

    hosd=i['host_os_dict']
    tosd=i['target_os_dict']

    # Check platform
    hplat=hosd.get('ck_name','')

    hproc=hosd.get('processor','')
    tproc=tosd.get('processor','')

    phosd=hosd.get('ck_name','')

    sv1='$('
    sv2=')'

    svarb=hosd.get('env_var_start','')
    svarb1=hosd.get('env_var_extra1','')
    svare=hosd.get('env_var_stop','')
    svare1=hosd.get('env_var_extra2','')

    iv=i.get('interactive','')
    cus=i.get('customize',{})
    install_env=cus.get('install_env',{})
    cfg=i.get('cfg',{})
    deps=i.get('deps',{})

    p=i.get('path','')
    pi=i.get('install_path','')

    # Set default parameters
    params={
      "python_bin_path":"/usr/bin/python",
      "tf_need_gcp":0,
      "tf_need_cuda":0,
      "gcc_host_compiler_path":"/usr/bin/gcc",
      "tf_need_hdfs":0
    }

    # Update params 
    params.update(cus.get('params',{}))
    
    # Get versions of CUDA and cuDNN in GPU enabled versions
    # NEED CK ENV UPDATE IN CUDNN. To get cudnn version from path.
    if install_env['USE_CUDA']:
        cuda_path = deps["compiler.cuda"]["dict"]["env"]["CK_ENV_COMPILER_CUDA"]
        if cuda_path is None:
            print "Error: CUDA dependence was not added."
            return 1
        params['tf_cuda_version'] = cuda_path[-3:]
        params['cuda_toolkit_path'] = cuda_path[:-4]
        
        #cuDNN Version from path here.
        
    # Load export-variables.template
    pp=os.path.join(p, 'export-variables.template')
    r=ck.load_text_file({'text_file':pp})
    if r['return']>0: return r

    s=r['string']

    # Replace all params
    for k in params:
        v=params[k]
        s=s.replace('$#'+k+'#$', str(v))

    # Record Makefile.config
    pp=os.path.join(p, 'export-variables')
    r=ck.save_text_file({'text_file':pp, 'string':s})
    if r['return']>0: return r

    # Update install environment, if needed
    ie={}

    return {'return':0, 'install_env':ie}
