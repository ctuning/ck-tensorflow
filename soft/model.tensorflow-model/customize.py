#
# Collective Knowledge (individual environment - setup)
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
#
# Developer: Zaborovskiy Vladislav, vladzab@yandex.ru
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

              host_os_uoa      - host OS UOApatch code
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

    winh=hosd.get('windows_base','')

    env=i['env']
    ep=cus['env_prefix']

    p1=os.path.dirname(fp)
    pl=os.path.dirname(p1)

    env[ep+'_ROOT']=pl
    env[ep+'_MODEL']=p1
    env[ep+'_PIPELINE_NAME']=os.path.basename(fp)

    install_env = cus.get('install_env', {})

    # Init common variables, they are set for all models
    if 'FROZEN_GRAPH' in install_env:
      env[ep+'_FROZEN_GRAPH']=os.path.join(p1, install_env['FROZEN_GRAPH'])
    if 'WEIGHTS_FILE' in install_env:
      env[ep+'_WEIGHTS']=os.path.join(p1, install_env['WEIGHTS_FILE'])
    if 'MODULE_FILE' in install_env:
      env[ep+'_MODULE']=os.path.join(p1, install_env['MODULE_FILE'])
    if 'LABELS_FILE' in install_env:
      env[ep+'_LABELS']=os.path.join(p1, install_env['LABELS_FILE'])

    # Init model specific variables
    # They should be started with MODEL_ prefix e.g. MODEL_MOBILENET_RESOLUTION
    # This prefix will be cut off as it already contained in cus['env_prefix']
    # so we'll get vars like CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION
    for varname in install_env.keys():
      if varname.startswith('MODEL_'):
        env[ep+varname[len('MODEL'):]] = install_env[varname]

    if 'DATASET_NAME' in install_env:
      env['CK_ENV_MODEL_DATASET_TYPE'] = install_env['DATASET_NAME']

    return {'return':0, 'bat':s}
