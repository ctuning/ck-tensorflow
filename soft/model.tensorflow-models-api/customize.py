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

    env = i['env']
    cus = i.get('customize',{})
    ep = cus['env_prefix']

    full_path = cus.get('full_path','')
    object_detection_dir    = os.path.dirname(full_path)
    research_models_dir     = os.path.dirname(object_detection_dir)
    slim_models_dir         = os.path.join(research_models_dir, 'slim')

    env[ep] = research_models_dir
    env[ep + '_RESEARCH']       = research_models_dir
    env[ep + '_SLIM']           = slim_models_dir
    env[ep + '_OBJ_DET_DIR']    = object_detection_dir

    env['PYTHONPATH'] = slim_models_dir + os.pathsep + \
                        research_models_dir + os.pathsep + \
                        '$PYTHONPATH'

    return {'return': 0, 'bat': ''}
