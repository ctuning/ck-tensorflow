## Install dependencies on Ubuntu

https://www.tensorflow.org/install/install_sources

### Install Python 3 (recommended) or Python 2 
```bash
$ sudo apt-get install libcupti-dev
$ sudo apt-get install python-dev python-pip python-wheel
```

### Install Python packages (in user-space)
```bash
python -m pip install google-pasta --user
python -m pip install opt-einsum --user
python -m pip install grpcio --user
python -m pip install protobuf --user
python -m pip install absl-py --user
python -m pip install wrapt --user
python -m pip install astor --user
python -m pip install termcolor --user
python -m pip install gast==0.2.2 --user
python -m pip install tensorboard==1.15.0 --user
python -m pip install tensorflow-estimator==1.15.1 --user
python -m pip install keras_applications==1.0.8 --no-deps --user
python -m pip install keras_preprocessing==1.1.0 --no-deps --user
```

## Prevent running out of memory

To prevent running out of memory during a build, restrict the build to use
e.g. 1 processor:
```bash
$ ck install package:lib-tensorflow-1.15.2-src-cuda --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=1
```

## Known problems

### CUDA 10.2 in not supported

Either [use 10.1](https://github.com/tensorflow/tensorflow/issues/34429) or
[patch](https://github.com/tensorflow/tensorflow/commit/67edc16326d6328e7ef096e1b06f81dae1bfb816).
