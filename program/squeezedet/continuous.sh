#! /bin/bash
ROOT=$CK_ENV_MODEL_SQUEEZEDET_ROOT

OUT_DIR="$PWD/out"
rm -rf "$OUT_DIR"

export TF_CPP_MIN_LOG_LEVEL=3 # supress TF debug output

IMAGES=$CK_ENV_DATASET_KITTI_IMAGE_DIR
LABELS=$CK_ENV_DATASET_KITTI_LABELS_DIR
CHECKPOINT="$ROOT/data/model_checkpoints/squeezeDet/model.ckpt-87000" # TODO: change this

${CK_ENV_COMPILER_PYTHON_FILE} "../continuous.py" --image_dir="$IMAGES" --label_dir="$LABELS" --out_dir="$OUT_DIR" --checkpoint="$CHECKPOINT"
