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

## Dependencies required for program running

Use `pip2.7` or `pip3.5` instead of just `pip`
when you have several python version installed
and program is running using not default python.

```
$ sudo pip install mock

# for python 2.7
$ sudo pip2.7 install enum34
```