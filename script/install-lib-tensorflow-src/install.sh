#! /bin/bash

#
# Installation script for the TensorFlow library.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#

function stage() {
  echo; echo "--------------------------------"; echo $1; echo
}

function exit_if_error() {
 if [ "${?}" != "0" ]; then exit 1; fi
}

function remove_dir_if_exists() {
  if [ -d $1 ]; then rm -rdf $1; fi
}

if [ "${PACKAGE_GIT}" == "YES" ] ; then
  stage "Cloning package ${PACKAGE_URL}"
  remove_dir_if_exists ${PACKAGE_SUB_DIR}
  git clone ${PACKAGE_GIT_CLONE_FLAGS} ${PACKAGE_URL} ${PACKAGE_SUB_DIR}
  exit_if_error

  stage "Checking out branch ${PACKAGE_GIT_CHECKOUT}"
  cd ${PACKAGE_SUB_DIR}
  git checkout ${PACKAGE_GIT_CHECKOUT}
  exit_if_error
fi

############################################################
if [ "${PACKAGE_PATCH}" == "YES" ] ; then
  if [ -d ${ORIGINAL_PACKAGE_DIR}/patch.${CK_TARGET_OS_ID} ] ; then
    echo ""
    echo "Patching source directory ..."

    cd ${INSTALL_DIR}/${PACKAGE_SUB_DIR}

    for i in ${ORIGINAL_PACKAGE_DIR}/patch.${CK_TARGET_OS_ID}/*
    do
      echo "$i"
      patch -p1 < $i

      if [ "${?}" != "0" ] ; then
        echo "Error: patching failed!"
        exit 1
      fi
    done
  fi
fi

stage "Configure environment variables"
# TF configure.py script will ask for all unset variables
export CC_OPT_FLAGS="-march=native"
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_S3=0
export TF_NEED_GDR=0
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_NEED_JEMALLOC=0
export TF_NEED_VERBS=0
export TF_NEED_MKL=0
export TF_DOWNLOAD_MKL=0
export TF_NEED_MPI=0
export TF_CUDA_CLANG=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_SET_ANDROID_WORKSPACE=0
export PYTHON_BIN_PATH=${CK_ENV_COMPILER_PYTHON_FILE}
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"

if [ "$CK_TF_NEED_CUDA" == "YES" ] ; then  
  export TF_NEED_CUDA=1
  echo "CUDA enabled"
else
  export TF_NEED_CUDA=0
  echo "CUDA disabled"
fi  

if [ "$CK_TF_NEED_TENSORRT" == "YES" ] ; then  
  export TF_NEED_TENSORRT=1
  echo "TENSORRT enabled"
else
  export TF_NEED_TENSORRT=0
  echo "TENSORRT disabled"
fi  

if [ "$CK_TF_ENABLE_XLA" == "YES" ] ; then
  export TF_ENABLE_XLA=1
  echo "XLA enabled"
else
  export TF_ENABLE_XLA=0
  echo "XLA disabled"
fi

if [ ${TF_NEED_CUDA} == 1 ] ; then
  stage "Configure environment variables for CUDA"
  CUDA_CONFIG_OPTS="--config=cuda "
  export TF_CUDA_COMPUTE_CAPABILITIES="3.5,5.2,6.1,6.2"
  export CUDA_TOOLKIT_PATH=${CK_ENV_COMPILER_CUDA}
  export CUDNN_INSTALL_PATH=${CK_ENV_LIB_CUDNN}
  export TF_CUDA_VERSION="$($CUDA_TOOLKIT_PATH/bin/nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')"
  export TF_CUDNN_VERSION="$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' $CUDNN_INSTALL_PATH/include/cudnn.h)"
  export GCC_HOST_COMPILER_PATH=${CK_ENV_COMPILER_GCC}/bin/${CK_CC}
fi

TARGET_OBJ_DIR=${INSTALL_DIR}/obj
TARGET_LIB_DIR=${INSTALL_DIR}/lib

stage "Run configuration script"
cd ${INSTALL_DIR}/src
./configure
exit_if_error

#hack to support tf14 build. not really elegant, however removes nvlink errors in nccl  : entry function '' with max regcount of 80 calls function '' with regcount of 96
sed -i 's/maxrregcount=96/maxrregcount=80/g' third_party/nccl/build_defs.bzl.tpl


stage "Build with bazel"

if [ -n "${CK_BAZEL_CACHE_DIR}" ]; then
  OUTPUT_USER_ROOT="--output_user_root ${CK_BAZEL_CACHE_DIR}"
fi

bazel \
  ${OUTPUT_USER_ROOT} \
  build \
  --config=opt \
  ${CUDA_CONFIG_OPTS} \
  ${OTHER_CONFIG_OPTS} \
  --jobs ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} \
  //tensorflow/tools/pip_package:build_pip_package
  
bazel shutdown
exit_if_error
# Seems bazel does not set error code when build process is interrupted 
# with Ctrl+C, so may be we need some other check for this 'error'

stage "Build pip package"
remove_dir_if_exists ${TARGET_OBJ_DIR}
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${TARGET_OBJ_DIR}
exit_if_error

stage "Install pip package"
# There should be only one whl file in the obj dir; it does not matter how it is called.
for WHL in ${TARGET_OBJ_DIR}/*.whl ; do
  echo "Processing ${WHL}"
  unzip -qo ${WHL} -d ${TARGET_OBJ_DIR}
  exit_if_error
done

remove_dir_if_exists ${TARGET_LIB_DIR}
mkdir ${TARGET_LIB_DIR}
if [ -d "${TARGET_OBJ_DIR}/tensorflow/" ] && [ -d "${TARGET_OBJ_DIR}/tensorflow_core/" ];
then # >= v1.15
  mv "${TARGET_OBJ_DIR}/tensorflow/"      ${TARGET_LIB_DIR}
  mv "${TARGET_OBJ_DIR}/tensorflow_core/" ${TARGET_LIB_DIR}
else # < v.1.15
  mv ${TARGET_OBJ_DIR}/tensorflow*data/purelib/* ${TARGET_LIB_DIR}
fi
rm -rdf ${TARGET_OBJ_DIR}/tensorflow-*.data
rm -rdf ${TARGET_OBJ_DIR}/tensorflow-*.dist-info

return 0
