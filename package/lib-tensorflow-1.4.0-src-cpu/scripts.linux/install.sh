#! /bin/bash

#
# Extra installation script.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#

# Configure environment variables
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

cd ${INSTALL_DIR}/src

echo "--------------------------------"
echo "Run TensorFlow configuration"
./configure

echo "--------------------------------"
echo "Building with bazel"
bazel build --config=opt --jobs ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} //tensorflow/tools/pip_package:build_pip_package
bazel shutdown

echo "--------------------------------"
echo "Building pip package"
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${TARGET_OBJ_DIR}

echo "--------------------------------"
echo "Installing pip package"
# There is only one whl file in the obj dir and we do not care its name
for WHL in ${TARGET_OBJ_DIR}/*.whl ; do
  echo "Processing ${WHL}"
  unzip -qo ${WHL} -d ${TARGET_OBJ_DIR}
done

if [ -d $TARGET_LIB_DIR ] ; then
  rm -rdf ${TARGET_LIB_DIR}
fi
mkdir ${TARGET_LIB_DIR}
mv ${TARGET_OBJ_DIR}/tensorflow*data/purelib/* ${TARGET_LIB_DIR}
rm -rdf ${TARGET_OBJ_DIR}/tensorflow-*.data
rm -rdf ${TARGET_OBJ_DIR}/tensorflow-*.dist-info

return 0
