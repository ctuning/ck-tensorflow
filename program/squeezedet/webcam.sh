#! /bin/bash
OUT_DIR="$PWD/out"
rm -rf "$OUT_DIR"

export PYTHONPATH="$PYTHONPATH:$CK_ENV_DEMO_SQUEEZEDET_SRC"
export TF_CPP_MIN_LOG_LEVEL=3 # supress TF debug output

CHECKPOINT=$CK_ENV_MODEL_SQUEEZEDET_MODEL
NET=$CK_ENV_MODEL_SQUEEZEDET_ID
DEVICE=${IMAGE_SOURCE_DEVICE:-0}

${CK_ENV_COMPILER_PYTHON_FILE} "../continuous.py" --out_dir="$OUT_DIR" --checkpoint="$CHECKPOINT" --demo_net="$NET" --finisher_file="$FINISHER_FILE" --input_device=$DEVICE
