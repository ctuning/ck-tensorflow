#! /bin/bash

#
# Installation script for RensorFlow library.
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

stage "Configure environment variables"
# TF configure.py script will ask for all unset variables
export CC_OPT_FLAGS="-march=native"
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
export PYTHON_BIN_PATH=${CK_ENV_COMPILER_PYTHON_FILE}
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"

if [ "$CK_TF_NEED_CUDA" == "YES" ] ; then  
  export TF_NEED_CUDA=1
  echo "CUDA enabled"
else
  export TF_NEED_CUDA=0
  echo "CUDA disabled"
fi  

if [ "$CK_TF_ENABLE_XLA" == "YES" ] ; then
  export TF_ENABLE_XLA=1
  echo "XLA enabled"
else
  export TF_ENABLE_XLA=0
  echo "XLA disabled"
fi

TARGET_OBJ_DIR=${INSTALL_DIR}/obj
TARGET_LIB_DIR=${INSTALL_DIR}/lib

stage "Run configuration script"
cd ${INSTALL_DIR}/src
./configure
exit_if_error

stage "Build with bazel"
bazel build --config=opt --jobs ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} //tensorflow/tools/pip_package:build_pip_package
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
mv ${TARGET_OBJ_DIR}/tensorflow*data/purelib/* ${TARGET_LIB_DIR}
rm -rdf ${TARGET_OBJ_DIR}/tensorflow-*.data
rm -rdf ${TARGET_OBJ_DIR}/tensorflow-*.dist-info

return 0
