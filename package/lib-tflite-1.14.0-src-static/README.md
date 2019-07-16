# TensorFlow Lite package

This package installs TensorFlow Lite from the official tagged release `1.14.0` of TensorFlow ("Heavy").

```bash
$ ck install package:lib-tflite-1.14.0-src-static --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=4
```

## Resolved issues
- https://github.com/tensorflow/tensorflow/issues/28926 (-flax-vector-conversion)
- https://github.com/tensorflow/tensorflow/issues/26731 (error: x29 cannot be used in asm here)
- https://github.com/tensorflow/tensorflow/issues/26731#issuecomment-499382957 (undefined reference to `flatbuffers::ClassicLocale::instance_')
- https://stackoverflow.com/questions/48410966/undefined-reference-to-shm-open (undefined reference to `shm_open')

## Unresolved issues
None.
