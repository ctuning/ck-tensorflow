#
# Collective Knowledge (individual environment - setup)
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
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

    # Get variables
    cus = i.get('customize',{})

    env = i['env']
    ep = cus['env_prefix']

    full_path = cus.get('full_path','')
    install_dir, model_filename = os.path.split(full_path)
    
    
    install_env = cus.get('install_env', {})

    # Init common variables, they are expected to be set for all models
    env[ep+'_MODEL_NAME'] = install_env['MODEL_NAME']
    env[ep+'_DEFAULT_HEIGHT'] = install_env['DEFAULT_HEIGHT']
    env[ep+'_DEFAULT_WIDTH'] = install_env['DEFAULT_WIDTH']
    env[ep+'_DATASET_TYPE'] = install_env['DATASET_TYPE']
    env[ep+'_LABELMAP_FILE'] = os.path.join(install_dir, install_env['LABELMAP_FILE'])

    frozen_graph_name = os.path.join(install_dir, install_env['FROZEN_GRAPH']) if 'FROZEN_GRAPH' in install_env else full_path
    env[ep+'_FROZEN_GRAPH'] = frozen_graph_name
    env[ep+'_TF_FROZEN_FILEPATH'] = frozen_graph_name  # the same (for compatibility)

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

    if 'WEIGHTS_FILE' in install_dir:   # optional
      env[ep+'_WEIGHTS_FILE'] = os.path.join(install_dir, install_env['WEIGHTS_FILE'])

    hosd=i['host_os_dict']
    winh=hosd.get('windows_base','')
    env['PYTHONPATH'] = install_dir + ( ';%PYTHONPATH%' if winh=='yes' else ':${PYTHONPATH}')

    return {'return': 0, 'bat': ''}
