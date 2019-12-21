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


##############################################################################

def version_cmd(i):
    import sys
    import json

    path_with_init_py       = i['full_path']                            # the full_path that ends with PACKAGE_NAME/__init__.py
    path_without_init_py    = os.path.dirname( path_with_init_py )
    package_name            = os.path.basename( path_without_init_py )
    site_dir                = os.path.dirname( path_without_init_py )
    ck                      = i['ck_kernel']
    cus                     = i['customize']

    detect_version_as           = str(cus.get('detect_version_as', ''))
    detect_version_externally   = cus.get('detect_version_externally', 'no') == 'yes'
    version_variable_name       = cus.get('version_variable_name', '__version__')

    desired_python_path     = i.get('deps', {}).get('python', {}).get('dict', {}).get('env', {}).get('CK_ENV_COMPILER_PYTHON_FILE', sys.executable)

    version_cmd     = '{} -c "import sys; sys.path.insert(0,\'{}\'); import {}; print({}.{})" >$#filename#$'.format(
        desired_python_path, site_dir, package_name, package_name, version_variable_name);

    print (version_cmd)

    return {'return':0, 'cmd':version_cmd}

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

    winh=hosd.get('windows_base','')

    env=i['env']
    ep=cus['env_prefix']

    p1=os.path.dirname(fp)
    lib_path        = os.path.dirname(p1)
    install_path    = os.path.dirname(lib_path )
    bin_path        = os.path.join(install_path , 'python_deps_site', 'bin')

    env[ep]         = install_path
    env[ep+'_LIB']  = lib_path
    env[ep+'_BIN']  = bin_path

    # Path to bundled protobuf.
    pb=os.path.join(lib_path ,'external','protobuf_archive','python')

    if winh=='yes':
        s+='\nset PYTHONPATH='+lib_path +';'+pb+';%PYTHONPATH%\n'
    else:
        s+='\nexport PYTHONPATH='+lib_path +':'+pb+':${PYTHONPATH}\nexport PATH='+bin_path+':${PATH}\n'

    return {'return':0, 'bat':s}
