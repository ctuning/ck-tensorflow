set PYTHONPATH=%PYTHONPATH%;%CK_ENV_DEMO_SQUEEZEDET_SRC%

rem supress TF debug output
set TF_CPP_MIN_LOG_LEVEL=3

set CHECKPOINT=%CK_ENV_MODEL_SQUEEZEDET_MODEL%
set NET=%CK_ENV_MODEL_SQUEEZEDET_ID%
set INPUT=%CK_ENV_DEMO_SQUEEZEDET_ROOT%/data/sample.png

"%CK_ENV_COMPILER_PYTHON_FILE%" "..\image_benchmark.py" --input_file="%INPUT%" --checkpoint="%CHECKPOINT%" --demo_net="%NET%"
