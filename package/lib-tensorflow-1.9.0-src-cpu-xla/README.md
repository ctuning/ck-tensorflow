## Install dependencies on Ubuntu

https://www.tensorflow.org/install/install_sources

```
$ sudo apt-get install python-numpy python-dev python-pip python-wheel
$ sudo apt-get install libcupti-dev
```

## Prevent running out of memory

To prevent running out of memory during a build, restrict the build to use
e.g. 1 processor:

```
$ ck install package:lib-tensorflow-1.9.0-src-cpu-xla --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=1
```
