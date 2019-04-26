## Install dependencies on Ubuntu

https://www.tensorflow.org/install/install_sources

### Install Python 2 or Python 3 (recommended)
```
$ sudo apt-get install libcupti-dev
$ sudo apt-get install python-dev python-pip python-wheel

### Install Python packages (in user-space)
```
$ python -m pip install numpy --user
$ python -m pip install gast --user
$ python -m pip install astor --user
$ python -m pip install termcolor --user
$ python -m pip install tensorflow-estimator==1.13.0 --user
$ python -m pip install keras_applications==1.0.4 --no-deps --user
$ python -m pip install keras_preprocessing==1.0.2 --no-deps --user
```

## Prevent running out of memory

To prevent running out of memory during a build, restrict the build to use
e.g. 1 processor:

```
$ ck install package:lib-tensorflow-1.13.1-src-cpu --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=1
```
