# Install static TensorFlow from source

This script builds a static TensorFlow library using scripts provided in the
`${CK-TOOLS}/lib-tensorflow-*-src-static/src/tensorflow/contrib/makefile/` directory.

For example, to build TensorFlow v1.13.1 for Android API 23 ("Marshmallow") using 2 CPU cores:
```
$ ck install package:lib-tensorflow-1.13.1-src-static --target_os=android24-arm64 \
--env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=2
```
