#! /bin/bash

set -e

${CK_MAKE} -j ${CK_HOST_CPU_NUMBER_OF_PROCESSORS:-1} tf_python_build_pip_package
