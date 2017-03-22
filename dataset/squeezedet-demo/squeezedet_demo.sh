#! /bin/bash
echo "Detecting objects in '${CK_ENV_MODEL_SQUEEZEDET_ROOT}/data/sample.png' ..."
cd ${CK_ENV_MODEL_SQUEEZEDET_ROOT} && ${CK_ENV_COMPILER_PYTHON_FILE} ./src/demo.py
echo "Check the output in '${CK_ENV_MODEL_SQUEEZEDET_ROOT}/data/out/' ..."
