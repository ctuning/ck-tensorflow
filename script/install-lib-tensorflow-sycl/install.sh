#! /bin/bash

#
# Installation script for the TensorFlow library, a SYCL implementation.
# https://developer.codeplay.com/computecppce/latest/getting-started-with-tensorflow
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

function patch_package() {
  if [ -d $1 ] ; then
    for i in $1/*.patch
    do
      if [ -f $i ]; then
        echo "Patch: $i"
        patch -p1 < $i
        exit_if_error
      fi
    done
  fi
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

if [ "${PACKAGE_PATCH}" == "YES" ] ; then
  stage "Patching source directory ..."
  cd ${INSTALL_DIR}/${PACKAGE_SUB_DIR}
  patch_package ${ORIGINAL_PACKAGE_DIR}
  patch_package ${ORIGINAL_PACKAGE_DIR}/patch.${CK_TARGET_OS_ID}
fi 

stage "Configure environment variables"
# TF configure.py script will ask for all unset variables
export CC_OPT_FLAGS="-march=native"
export TF_NEED_S3=0
export TF_NEED_GDR=0
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_JEMALLOC=0
export TF_NEED_VERBS=0
export TF_NEED_MKL=0
export TF_DOWNLOAD_MKL=0
export TF_NEED_MPI=0
export TF_CUDA_CLANG=0
export TF_NEED_CUDA=0
export TF_ENABLE_XLA=0
export TF_NEED_KAFKA=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_VECTORIZE_SYCL=0
export PYTHON_BIN_PATH=${CK_ENV_COMPILER_PYTHON_FILE}
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')" 

if [ "$CK_TF_NEED_OPENCL" == "YES" ] ; then  
  echo "OpenCL enabled"
  export TF_NEED_OPENCL=1
  export TF_NEED_OPENCL_SYCL=1
  export TF_NEED_COMPUTECPP=1
  export COMPUTECPP_TOOLKIT_PATH=${CK_ENV_COMPILER_COMPUTECPP}
  BUILD_CONFIG_SYCL="--config=sycl"
else
  echo "OpenCL disabled"
  export TF_NEED_OPENCL=0
  export TF_NEED_OPENCL_SYCL=0
  export TF_NEED_COMPUTECPP=0
fi  

TARGET_OBJ_DIR=${INSTALL_DIR}/obj
TARGET_LIB_DIR=${INSTALL_DIR}/lib

if [ -n "${CK_BAZEL_CACHE_DIR}" ]; then
  OUTPUT_USER_ROOT="--output_user_root ${CK_BAZEL_CACHE_DIR}"
fi

stage "Run configuration script"
cd ${INSTALL_DIR}/src
./configure
exit_if_error

stage "Build with bazel"
bazel \
  ${OUTPUT_USER_ROOT} \
  build \
  --config=opt \
  ${BUILD_CONFIG_SYCL} \
  --jobs ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} \
  //tensorflow/tools/pip_package:build_pip_package

bazel shutdown
exit_if_error

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
mv ${TARGET_OBJ_DIR}/tensorflow*data/purelib/* ${TARGET_LIB_DIR}
rm -rdf ${TARGET_OBJ_DIR}/tensorflow-*.data
rm -rdf ${TARGET_OBJ_DIR}/tensorflow-*.dist-info

return 0
