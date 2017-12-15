#! /bin/bash

export PYTHONPATH="$PYTHONPATH:$CK_ENV_DEMO_SQUEEZEDET_SRC"
export TF_CPP_MIN_LOG_LEVEL=3 # supress TF debug output

CHECKPOINT=$CK_ENV_MODEL_SQUEEZEDET_MODEL
NET=$CK_ENV_MODEL_SQUEEZEDET_ID
INPUT=$CK_ENV_DEMO_SQUEEZEDET_ROOT/data/sample.png

${CK_ENV_COMPILER_PYTHON_FILE} "../image_benchmark.py" --input_file="$INPUT" --checkpoint="$CHECKPOINT" --demo_net="$NET" --batch_count=$BATCH_COUNT --batch_size=$BATCH_SIZE