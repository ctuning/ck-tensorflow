#
# Collective Knowledge (individual environment - setup)
#
# See CK LICENSE.txt for licensing details
#
# Developer: Zaborovskiy Vladislav, vladzab@yandex.ru, 
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
    p1=os.path.dirname(pi)
    p2=os.path.dirname(p1)
    p3=os.path.dirname(p2)
    dirname=os.path.basename(p2)

    ep=cus.get('env_prefix','')

    full_path = p2

    if cus.get('params',{}).get('min','')=='yes':
      full_path = p1
    elif dirname == 'data_object_image_2':
      full_path = p3
    image_dir =  cus.get('install_env', '').get('IMAGE_DIR', '')
    labels_dir =  cus.get('install_env', '').get('LABELS_DIR', '')

    full_images_path = os.path.join(full_path, image_dir)
    env[ep + "_IMAGE_DIR"] = full_images_path
    env['CK_ENV_DATASET_IMAGE_DIR'] = full_images_path
    if labels_dir:
      full_labels_path = os.path.join(full_path, labels_dir)
      env[ep + "_LABELS_DIR"] = full_labels_path
      env['CK_ENV_DATASET_LABELS_DIR'] = full_labels_path
      env['CK_ENV_DATASET_ANNOTATIONS'] = full_labels_path
    env[ep]=full_path
    env['CK_SQUEEZENET_KITTI']=full_path

    env['CK_ENV_DATASET_TYPE']='kitti'

    return {'return':0, 'bat':s}

