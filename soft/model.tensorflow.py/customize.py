#
# Collective Knowledge (individual environment - setup)
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
#
# Developer: Grigori Fursin, Grigori.Fursin@cTuning.org, http://fursin.net
#

import os

##############################################################################
# setup environment setup

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
            }

    Output: {
              return       - return code =  0, if successful
                                         >  0, if error
              (error)      - error text if return > 0

              bat          - prepared string for bat file
            }

    """

    import os

    # Get variables
    ck=i['ck_kernel']
    s=''

    iv=i.get('interactive','')

    cus=i.get('customize',{})
    fp=cus.get('full_path','')

    hosd=i['host_os_dict']
    tosd=i['target_os_dict']

    sdirs=hosd.get('dir_sep','')

    # Check platform
    hplat=hosd.get('ck_name','')

    hproc=hosd.get('processor','')
    tproc=tosd.get('processor','')

    remote=tosd.get('remote','')
    tbits=tosd.get('bits','')

    env=i['env']

    install_root    = os.path.dirname(fp)
    install_env     = cus.get('install_env', {})

    ep=cus['env_prefix']

    # Provide the installation root where all files live:
    env[ep + '_ROOT'] = install_root

    # Omit the trivial preprocessing step by detecting the necessary model files during installation
    #
    # These automatically detected values take lower precedence and can be overridden by values in install_env
    #
    for filename in os.listdir(install_root):
        filepath = os.path.join(install_root, filename)
        if filename.endswith('.pb'):
            env[ep + '_TF_FROZEN_FILENAME'] = filename
            env[ep + '_TF_FROZEN_FILEPATH'] = filepath
        elif filename.endswith('.tflite'):
            env[ep + '_TFLITE_FILENAME'] = filename
            env[ep + '_TFLITE_FILEPATH'] = filepath
        elif filename.endswith('_info.txt'):
            # Read input and output layer names from graph info file
            with open(filepath, 'r') as f:
                for line in f:
                    line_parts = line.split(' ')
                    if len(line_parts) == 2:
                        if line_parts[0] == 'Input:':
                            env[ep + '_INPUT_LAYER_NAME'] = line_parts[1].strip()
                        elif line_parts[0] == 'Output:':
                            env[ep + '_OUTPUT_LAYER_NAME'] = line_parts[1].strip()

    # Init common variables, they are set for all models:
    #
    # This group should end with _FILE prefix e.g. TFLITE_FILE
    # This suffix will be cut off and prefixed by cus['env_prefix']
    # so we'll get vars like CK_ENV_TENSORFLOW_MODEL_TFLITE
    for varname in install_env.keys():
        if varname.endswith('_FILE'):
            env[ep + '_' + varname[:-len('_FILE')]] = os.path.join(install_root, install_env[varname])

    # Init model-specific variables:
    #
    # This other group should be started with MODEL_ prefix e.g. MODEL_MOBILENET_RESOLUTION
    # This prefix will be cut off as it already contained in cus['env_prefix']
    # so we'll get vars like CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION
    for varname in install_env.keys():
        if varname.startswith('MODEL_'):
            env[ep+varname[len('MODEL'):]] = install_env[varname]

    # Just copy those without any change in the name:
    #
    for varname in install_env.keys():
        if varname.startswith('ML_MODEL_'):
            env[varname] = install_env[varname]
    
    return {'return':0, 'bat':s}
