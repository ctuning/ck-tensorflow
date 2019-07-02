# TensorFlow Lite package

The package is built from official tagged release 1.14.0 of 'big' TensorFlow.

```bash
$ ck install package:lib-tflite-1.14.0-src-static
```

## Resolved issues
- https://github.com/tensorflow/tensorflow/issues/28926 (-flax-vector-conversion)
- https://github.com/tensorflow/tensorflow/issues/26731 (error: x29 cannot be used in asm here)

## Unresolved issues
- https://github.com/tensorflow/tensorflow/issues/26731#issuecomment-499382957 (undefined reference to `flatbuffers::ClassicLocale::instance_')
