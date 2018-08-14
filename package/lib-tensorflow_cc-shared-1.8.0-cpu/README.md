## Install dependencies on Ubuntu

```
$ sudo apt-get install software-properties-common
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo apt-get update
$ sudo apt-get install build-essential curl git cmake unzip autoconf autogen libtool mlocate zlib1g-dev \
                     g++-6 python python3-numpy python3-dev python3-pip python3-wheel wget
$ sudo updatedb

```

## Prevent running out of memory

To prevent running out of memory during a build, restrict the build to use
e.g. 2 processors:
```
$ ck install package:lib-tensorflow_cc-shared-1.8.0-cpu --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=2
```