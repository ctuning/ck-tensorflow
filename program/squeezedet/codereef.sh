#! /bin/bash
OUT_DIR="$PWD/out"

if [ -z "$SKIP_FILES_INCLUDING" ]; then
    rm -rf "$OUT_DIR"
fi

export PYTHONPATH="$PYTHONPATH:$CK_ENV_DEMO_SQUEEZEDET_SRC"
export TF_CPP_MIN_LOG_LEVEL=3 # supress TF debug output

IMAGES=$CK_ENV_DATASET_IMAGE_DIR
LABELS=$CK_ENV_DATASET_LABELS_DIR

CHECKPOINT=$CK_ENV_MODEL_SQUEEZEDET_MODEL
NET=$CK_ENV_MODEL_SQUEEZEDET_ID

export CODEREEF="YES"
                     
${CK_ENV_COMPILER_PYTHON_FILE} "../continuous.py" --image_dir="$OUT_DIR" --label_dir="$LABELS" --out_dir="$OUT_DIR" --checkpoint="$CHECKPOINT" --demo_net="$NET" --finisher_file="$FINISHER_FILE" --skip_files_including="$SKIP_FILES_INCLUDING"
