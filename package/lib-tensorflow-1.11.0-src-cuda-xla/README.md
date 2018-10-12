## Install dependencies on Ubuntu

https://www.tensorflow.org/install/install_sources

```
$ sudo apt-get install python-numpy python-dev python-pip python-wheel
$ sudo apt-get install libcupti-dev
$ pip install keras_applications==1.0.4 --no-deps
$ pip install keras_preprocessing==1.0.2 --no-deps
```

## Install Python libraries on aarch64-architecture
```
$ pip install gast
$ pip install astor
$ pip install termcolor
```

## Prevent running out of memory

To prevent running out of memory during a build, restrict the build to use
e.g. 1 processor:

```
$ ck install package:lib-tensorflow-1.11.0-src-cuda-xla --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=1
```

**NB:** gcc 5.4 is required on Ubuntu 16.04, see [CUDA System requirements](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements). Even using of gcc 5.5 issues with errors similar to described [here](https://github.com/tensorflow/tensorflow/issues/10220) or [here](https://github.com/tensorflow/tensorflow/issues/18522).
