#
# CK configuration script for TensorFlow package
#
# Developer(s): 
#  * Vladislav Zaborovskiy, vladzab@yandex.ru
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
    s=''

    hosd=i['host_os_dict']
    tosd=i['target_os_dict']

    # Check platform
    hplat=hosd.get('ck_name','')

    hproc=hosd.get('processor','')
    tproc=tosd.get('processor','')

    phosd=hosd.get('ck_name','')

    svarb=hosd.get('env_var_start','')
    svarb1=hosd.get('env_var_extra1','')
    svare=hosd.get('env_var_stop','')
    svare1=hosd.get('env_var_extra2','')

    iv=i.get('interactive','')
    cus=i.get('customize',{})
    install_env=cus.get('install_env',{})
    cfg=i.get('cfg',{})
    deps=i.get('deps',{})
    ft=i.get('features',{})

    # Check python path and version
    python_ver=deps.get('python',{}).get('ver','')

    python3=0
    if python_ver.startswith('3'):
        python3=1

    python_path=deps.get('python',{}).get('dict',{}).get('customize',{}).get('full_path','')

    # Check GCC path
    gcc_path=deps.get('compiler.gcc',{}).get('dict',{}).get('customize',{}).get('full_path','')

    cxx_path=deps.get('compiler.gcc',{}).get('dict',{}).get('customize',{}).get('full_path','')
    cxx=deps.get('compiler.gcc',{}).get('dict',{}).get('env',{}).get('CK_CXX','')

    cxx_path=os.path.join(os.path.dirname(cxx_path),cxx)

    p=i.get('path','')
    pi=i.get('install_path','')

    # Set default parameters
    params={
      "python_bin_path":python_path,
      "gcc_host_compiler_path":gcc_path,
      "cxx_host_compiler":cxx_path,
      "python3":python3,
      "tf_need_gcp":0,
      "tf_need_cuda":0,
      "tf_need_opencl":0,
      "cuda_toolkit_path":"",
      "tf_cuda_version":"",
      "tf_cuda_compute_capabilities":"",
      "cudnn_install_path":"",
      "tf_cudnn_version":"",
      "tf_need_hdfs":0
    }

    # Update params 
    params.update(cus.get('params',{}))

    if params.get('tf_need_opencl',0)==1:
        ccpp=deps.get('compiler.computecpp',{}).get('dict',{}).get('env',{}).get('CK_ENV_COMPILER_COMPUTECPP','')

        params['computecpp_toolkit_path']=ccpp

    elif params.get('tf_need_cuda',0)==1:
        # Cuda compute capabilities
        cc=''
        cft=ft.get('gpgpu',[])
        if len(cft)>0:
            # For now first device
            cc=cft[0].get('gpgpu_misc',{}).get('gpu compute capability','')

        params['tf_cuda_compute_capabilities']=cc

        # Cuda path
        cuda_path=deps.get('compiler.cuda',{}).get('dict',{}).get('env',{}).get('CK_ENV_COMPILER_CUDA','')
        if cuda_path=='':
            return {'return':1, 'error':'CUDA dependence was not added'}

        cuda_ver=deps.get('compiler.cuda',{}).get('ver','')
        j=cuda_ver.find('.')
        if j>0:
            j=cuda_ver.find('.',j+1)
            if j>0:
                cuda_ver=cuda_ver[:j]

        params['tf_cuda_version'] = cuda_ver
        params['cuda_toolkit_path'] = cuda_path

        #cuDNN Version from path here.
        cudnn_path=deps.get('lib.cudnn',{}).get('dict',{}).get('env',{}).get('CK_ENV_LIB_CUDNN','')
        cudnn_ver=deps.get('lib.cudnn',{}).get('ver','')
        if cudnn_path=='':
            return {'return':1, 'error':'cuDNN dependence was not added'}
        if cudnn_ver.startswith('api-'):
            cudnn_ver=cudnn_ver[4:]

        params['cudnn_install_path']=cudnn_path
        params['tf_cudnn_version']=cudnn_ver

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
