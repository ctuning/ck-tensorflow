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

    import os

    # Get variables
    ck=i['ck_kernel']
    s=''

    iv=i.get('interactive','')

    cus=i.get('customize',{})
    full_path = cus.get('full_path','')

    hosd=i['host_os_dict']
    tosd=i['target_os_dict']

    winh=hosd.get('windows_base','')

    env=i['env']
    ep=cus['env_prefix']

    lib_dir = os.path.dirname(full_path)
    install_dir = os.path.dirname(lib_dir)
    src_dir = os.path.join(install_dir, 'src')

    target_os_dict = i.get('target_os_dict', {})
    target_os_name = target_os_dict.get('ck_name2', '')

    env[ep+'_LIBS_DIRS'] = '-L' + lib_dir
    if target_os_name == 'android':
      env[ep+'_LIBS'] = '-ltensorflow-lite -llog'
    elif target_os_name == 'linux':
      env[ep+'_LIBS'] = '-pthread -ltensorflow-lite -ldl -latomic -lrt'
    else:
      return {'return': -1, 'error': 'Unsupported target OS'}

    env[ep] = install_dir
    env[ep+'_LIB'] = lib_dir
    env[ep+'_INCLUDE0'] = src_dir
    env[ep+'_INCLUDE1'] = os.path.join(src_dir, 'tensorflow', 'contrib', 'lite', 'downloads', 'flatbuffers', 'include')

    return {'return': 0, 'bat': s}
