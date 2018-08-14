#!/bin/bash
set -e

# configure environmental variables
export CC_OPT_FLAGS=${CC_OPT_FLAGS:-"-march=haswell"}
export TF_NEED_GCP=${TF_NEED_GCP:-0}
export TF_NEED_HDFS=${TF_NEED_HDFS:-0}
export TF_NEED_OPENCL=${TF_NEED_OPENCL:-0}
export TF_NEED_OPENCL_SYCL=${TF_NEED_OPENCL_SYCL:-0}
export TF_NEED_TENSORRT=${TF_NEED_TENSORRT:-0}
export TF_NEED_JEMALLOC=${TF_NEED_JEMALLOC:-1}
export TF_NEED_VERBS=${TF_NEED_VERBS:-0}
export TF_NEED_MKL=${TF_NEED_MKL:-1}
export TF_DOWNLOAD_MKL=${TF_DOWNLOAD_MKL:-1}
export TF_NEED_MPI=${TF_NEED_MPI:-0}
export TF_ENABLE_XLA=${TF_ENABLE_XLA:-1}
export TF_NEED_S3=${TF_NEED_S3:-0}
export TF_NEED_GDR=${TF_NEED_GDR:-0}
export TF_CUDA_CLANG=${TF_CUDA_CLANG:-0}
export TF_SET_ANDROID_WORKSPACE=${TF_SET_ANDROID_WORKSPACE:-0}
export TF_NEED_KAFKA=${TF_NEED_KAFKA:-0}
export PYTHON_BIN_PATH=${CK_ENV_COMPILER_PYTHON_FILE}
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"

if [ "$CK_TF_NEED_CUDA" == "YES" ] ; then  
  export TF_NEED_CUDA=1
  echo "CUDA enabled"
else
  export TF_NEED_CUDA=0
  echo "CUDA disabled"
fi

# configure cuda environmental variables
if [ ${TF_NEED_CUDA} == 1 ]; then
  CUDA_CONFIG_OPTS="--config=opt --config=cuda"
  export TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES:-"3.5,5.2,6.1,6.2"}
  export CUDA_TOOLKIT_PATH=${CK_ENV_COMPILER_CUDA}
  export CUDNN_INSTALL_PATH=${CK_ENV_LIB_CUDNN}
  export TF_CUDA_VERSION="$($CUDA_TOOLKIT_PATH/bin/nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')"
  export TF_CUDNN_VERSION="$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' $CUDNN_INSTALL_PATH/include/cudnn.h)"
  export GCC_HOST_COMPILER_PATH=${CK_ENV_COMPILER_GCC}/bin/${CK_CC}
  export CLANG_CUDA_COMPILER_PATH=${CLANG_CUDA_COMPILER_PATH:-"/usr/bin/clang"}
  export TF_CUDA_CLANG=${TF_CUDA_CLANG:-0}
fi

if [ -n "${CK_BAZEL_CACHE_DIR}" ]; then
  OUTPUT_USER_ROOT="--output_user_root ${CK_BAZEL_CACHE_DIR}"
fi

# configure and build
./configure
bazel \
  ${OUTPUT_USER_ROOT} \
  build -c opt \
  ${CUDA_CONFIG_OPTS} \
  --config=monolithic \
  --jobs ${CK_HOST_CPU_NUMBER_OF_PROCESSORS} \
  tensorflow:libtensorflow_cc.so

bazel shutdown
