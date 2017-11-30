## Install dependencies on Ubuntu

```
$ sudo apt install swig3.0

```

## Prevent running out of memory

To prevent running out of memory during a build, restrict the build to use
e.g. 2 processors:
```
$ ck install package:lib-tensorflow-cmake --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=2
```
