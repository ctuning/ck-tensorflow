#! /bin/bash
ROOT=$CK_ENV_MODEL_SQUEEZEDET_ROOT
OUT_DIR="$PWD/out"
rm -rf "$OUT_DIR"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"
export TF_CPP_MIN_LOG_LEVEL=3 # supress TF debug outpput
${CK_ENV_COMPILER_PYTHON_FILE} "../continuous.py" --image_dir="$ROOT/data/KITTI/training/image_2" --label_dir="$ROOT/data/KITTI/training/label_2" --out_dir="$OUT_DIR" --checkpoint="$ROOT/data/model_checkpoints/squeezeDet/model.ckpt-87000"
