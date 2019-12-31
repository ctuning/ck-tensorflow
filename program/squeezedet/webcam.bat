set OUT_DIR=%cd%\out
del /Q /S -rf %OUT_DIR%

set PYTHONPATH=%PYTHONPATH%;%CK_ENV_DEMO_SQUEEZEDET_SRC%

rem supress TF debug output
set TF_CPP_MIN_LOG_LEVEL=3

set CHECKPOINT=%CK_ENV_MODEL_SQUEEZEDET_MODEL%
set NET=%CK_ENV_MODEL_SQUEEZEDET_ID%

if "%IMAGE_SOURCE_DEVICE%" == "" (
    set IMAGE_SOURCE_DEVICE=0
)

"%CK_ENV_COMPILER_PYTHON_FILE%" "..\continuous.py" --out_dir="%OUT_DIR%" --checkpoint="%CHECKPOINT%" --demo_net="%NET%" --finisher_file="%FINISHER_FILE%" --input_device=%IMAGE_SOURCE_DEVICE%
