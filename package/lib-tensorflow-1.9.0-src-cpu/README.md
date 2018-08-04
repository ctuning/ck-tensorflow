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
$ ck install package:lib-tensorflow-1.9.0-src-cpu --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=1
```

## Temporarily introduce patch

Fails both [without](https://github.com/tensorflow/tensorflow/issues/21332) and
[with](https://github.com/tensorflow/tensorflow/pull/16175#issuecomment-410437372)
the patch on the TX1. Committed to be tested on other aarch64 platforms.
