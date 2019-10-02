# TensorFlow Lite package

This package installs TensorFlow Lite from the official tagged release `1.15.0` of TensorFlow ("Heavy").

```bash
$ ck install package:lib-tflite-1.15.0-src-static --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=4
```

## Unresolved issues

### Cannot build for Android

Tried installing with `--target_os=android23-arm64` using Android NDK 17.2:
- GCC 4.9.x complained about paths to `stdint.h`.
- LLVM 6.0.2 complained about `string` without `std::`.

## Resolved issues
