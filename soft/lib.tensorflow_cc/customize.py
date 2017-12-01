#
# Collective Knowledge (individual environment - setup)
#
# See CK LICENSE.txt for licensing details
# See CK COPYRIGHT.txt for copyright details
#
# Developer: Grigori Fursin
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

    env=i['env']
    ep=cus['env_prefix']

    # $CK_TOOLS/install/lib/tensorflow_cc/libtensorflow_cc.so
    ff=os.path.basename(fp)
    # $CK_TOOLS/install/lib/tensorflow_cc/
    p0=os.path.dirname(fp)
    # $CK_TOOLS/install/lib/
    p1=os.path.dirname(p0)
    # $CK_TOOLS/install/
    p2=os.path.dirname(p1)
    # $CK_TOOLS/
    p3=os.path.dirname(p2)
    
    # Standard paths.
    env[ep]=p3
    cus['dynamic_lib']=ff
    cus['path_include']=os.path.join(p2,'include','tensorflow')
    # Append library path to $LIBRARY_PATH and $LD_LIBRARY_PATH.
    cus['path_lib']=p0 # os.path.join(p2,'lib','tensorflow_cc')
    r=ck.access({'action': 'lib_path_export_script', 'module_uoa': 'os', 'host_os_dict': hosd, 'lib_path': cus.get('path_lib','')})
    if r['return']>0: return r
    s+=r['script']
    # Path to libtensorflow_cc.so.
    env[ep+'_LIBTENSORFLOW_CC']=os.path.join(p2,'lib','tensorflow_cc','libtensorflow_cc.so') 
    # Path to libprotobuf.a.
    env[ep+'_LIBPROTOBUF']=os.path.join(p2,'lib','tensorflow_cc','libprotobuf.a')

    # Tensorflow 1.4 includes additional lib that need to be linked to programs
    if os.path.isfile(os.path.join(p0, 'libtensorflow_framework.so')):
        env[ep+'_LINK_EXTRA_LIBS']='-ltensorflow_framework'

    # Path to $CK_TOOLS/install/lib/cmake/TensorflowCC.
    env[ep+'_CMAKE']=os.path.join(p2,'cmake')
    # TensorFlow_CC-specific include paths from TensorflowCCSharedTargets.cmake.
    env[ep+'_INCLUDE0']=os.path.join(p2,'include','tensorflow')
    env[ep+'_INCLUDE1']=os.path.join(p2,'include','tensorflow','bazel-genfiles')
    env[ep+'_INCLUDE2']=os.path.join(p2,'include','tensorflow','tensorflow','contrib','makefile','downloads')
    env[ep+'_INCLUDE3']=os.path.join(p2,'include','tensorflow','tensorflow','contrib','makefile','downloads','eigen')
    env[ep+'_INCLUDE4']=os.path.join(p2,'include','tensorflow','tensorflow','contrib','makefile','downloads','gemmlowp')
    env[ep+'_INCLUDE5']=os.path.join(p2,'include','tensorflow','tensorflow','contrib','makefile','gen','protobuf-host','include')

    # Tensorflow 1.4 requires additional include path
    include6 = os.path.join(p2,'include','tensorflow','tensorflow','contrib','makefile','downloads','nsync','public')
    if os.path.isdir(include6):
        env[ep+'_INCLUDE6']=include6

    return {'return':0, 'bat':s}
