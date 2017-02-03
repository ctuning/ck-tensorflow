#! /bin/bash


export PATH=/home/fanran/CK/ck-env/platform.init/generic-linux-dummy:$PATH


. /home/fanran/CK/local/env/6c1b146ac0fa13ab/env.sh

export BATCH_SIZE=32
export NUM_BATCHES=25


echo    executing code ...
 ${CK_ENV_COMPILER_PYTHON_FILE} ../alexnet_benchmark.py --batch_size=${BATCH_SIZE} --num_batches=${NUM_BATCHES}
