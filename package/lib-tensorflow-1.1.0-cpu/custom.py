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

    hos=i['host_os_uoa']
    tos=i['target_os_uoa']

    hosd=i['host_os_dict']
    tosd=i['target_os_dict']

    hbits=hosd.get('bits','')
    tbits=tosd.get('bits','')

    hname=hosd.get('ck_name','')    # win, linux
    hname2=hosd.get('ck_name2','')  # win, mingw, linux, android
    macos=hosd.get('macos','')      # yes/no

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
    ie=cus.get('install_env',{})
    cfg=i.get('cfg',{})
    deps=i.get('deps',{})
    ft=i.get('features',{})

    # Check bits
    if hbits!='64':
       return {'return':1, 'error':'only 64-bit host is supported for this package'}

    # Check python path and version
    python_ver=deps.get('python',{}).get('ver','')
    spython_ver=deps.get('python',{}).get('dict',{}).get('setup',{}).get('version_split',[])

    python3=0
    if python_ver.startswith('3'):
        python3=1

    sver2=''
    if len(spython_ver)>1:
       sver2=spython_ver[1]

    # Check download path
    p='https://storage.googleapis.com/tensorflow/'

    if hname=='win':
       p+='windows'
    else:
       p+='linux'

    tp='cpu'
    tp1=''
    if ie.get('TF_CUDA','')=='YES':
       tp='gpu'
       tp1='_gpu'

    proto=p+'/cpu/'

    p+='/'+tp+'/'

    p+='tensorflow'+tp1+'-'+ie['TENSORFLOW_PACKAGE_VER']

    if macos=='yes':
       if python3==1:
          px='py2-none-any.whl'
       else:
          px='py3-none-any.whl'

    elif hname=='win':
       if python3==1:

          supported_python_ver2_on_win=cus.get('supported_python_ver2_on_win',[])

          if len(supported_python_ver2_on_win)>0 and sver2 not in supported_python_ver2_on_win:
             return {'return':1, 'error':'this package supports only Python 3.'+str(supported_python_ver2_on_win)+' on Windows'}

          px='cp3'+str(sver2)+'-cp3'+str(sver2)+'m-win_amd64.whl'
       else:
          return {'return':1, 'error':'Python 2 is not supported for this package on Windows'}

    else:
       if python3==1:
          px='cp34-cp34m-linux_x86_64.whl'
          if sver2==5:
             px='cp35-cp35m-linux_x86_64.whl'
          elif sver2==6:
             px='cp36-cp36m-linux_x86_64.whl'
       else:
          px='cp27-none-linux_x86_64.whl'

    p+='-'+px

    nie={'PYTHON3':python3,
         'TF_PYTHON_URL':p}

    # Prepare protobuf
    proto+='protobuf-3.1.0-'

    if hname=='win':
       proto='https://pypi.python.org/packages/b2/30/ab593c6ae73b45a5ef0b0af24908e8aec27f79efcda2e64a3df7af0b92a2/protobuf-3.1.0-py2.py3-none-any.whl'
    else:
       if python3==1:
          proto+='cp35-none-linux_x86_64.whl'
       else:
          proto+='cp27-none-linux_x86_64.whl'

    nie['PROTOBUF_PYTHON_URL']=proto

    return {'return':0, 'install_env':nie}
