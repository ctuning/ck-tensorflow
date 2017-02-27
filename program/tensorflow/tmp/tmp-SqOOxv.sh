#! /bin/bash


export PATH=/home/fanran/CK/ck-env/platform.init/generic-linux-dummy:$PATH


. /home/fanran/CK/local/env/eeb0257bdde63022/env.sh

export BATCH_SIZE=32
export NUM_BATCHES=25


export CK_DATASET_PATH=/home/fanran/CK/ck-tensorflow/dataset/benchmark-googlenet/

export CK_DATASET_FILENAME=benchmark-googlenet.py

echo    executing code ...
 ${CK_ENV_COMPILER_PYTHON_FILE} /home/fanran/CK/ck-tensorflow/dataset/benchmark-googlenet/benchmark-googlenet.py --batch_size=${BATCH_SIZE} --num_batches=${NUM_BATCHES}
