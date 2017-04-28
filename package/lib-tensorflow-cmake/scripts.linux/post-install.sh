#! /bin/bash

set -e

${CK_MAKE} tf_python_build_pip_package

${CK_ENV_COMPILER_PYTHON_FILE} -m pip install tf_python/dist/tensorflow-*.whl
