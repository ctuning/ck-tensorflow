#! /bin/bash
ROOT=$PWD/..

OUT_DIR="$PWD/out"
rm -rf "$OUT_DIR"

export PYTHONPATH="$PYTHONPATH:$ROOT/src"
export TF_CPP_MIN_LOG_LEVEL=3 # supress TF debug output

IMAGES=$CK_ENV_DATASET_KITTI_IMAGE_DIR
LABELS=$CK_ENV_DATASET_KITTI_LABELS_DIR

CHECKPOINT=$CK_ENV_MODEL_SQUEEZEDET_MODEL

${CK_ENV_COMPILER_PYTHON_FILE} "../continuous.py" --image_dir="$IMAGES" --label_dir="$LABELS" --out_dir="$OUT_DIR" --checkpoint="$CHECKPOINT"
