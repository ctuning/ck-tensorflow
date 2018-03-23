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

    pi=os.path.dirname(fp)

    install_env = cus.get('install_env', {})

    ep=cus['env_prefix']

    # Init common variables, they are set for all models
    env[ep+'_WEIGHTS']=os.path.join(pi, install_env['WEIGHTS_FILE'])
    env[ep+'_MODULE']=os.path.join(pi, install_env['MODULE_FILE'])

    # Init model specific variables
    # They should be started with MODEL_ prefix e.g. MODEL_MOBILENET_RESOLUTION
    # This prefix will be cut off as it already contained in cus['env_prefix']
    # so we'll get vars like CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION
    for varname in install_env.keys():
      if varname.startswith('MODEL_'):
        env[ep+varname[len('MODEL'):]] = install_env[varname]
    
    return {'return':0, 'bat':s}
