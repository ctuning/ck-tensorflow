# TensorFlow Lite package

This package installs TensorFlow Lite from the official tagged release `1.15.0` of TensorFlow ("Heavy").

**NB:** This package enables the new GEMM backend of TFLite.
To reproduce [MLPerf Inference v0.5 results](https://github.com/mlperf/inference_results_v0.5)
from [dividiti](http://dividiti.com), use `ck-mlperf:package:lib-tflite-1.15.0-rc2-src-static`.

```bash
$ ck install package:lib-tflite-1.15.0-src-static --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=4
```

**NB:** To use machine-specific build options (very important on Raspberry Pi, for example!), use:
```
$ ck install package:lib-tflite-1.15.0-src-static \
--env.EXTRA_CXXFLAGS="-march=armv7-a+neon+vfpv4 -mfpu=neon-vfpv4"
```

## Unresolved issues

### Cannot build for Android

Tried installing with `--target_os=android23-arm64` using Android NDK 17.2:
- GCC 4.9.x complained about paths to `stdint.h`.
- LLVM 6.0.2 failed when linking:
```
/home/anton/CK_TOOLS/lib-tflite-src-static-1.15.0-rc2-llvm-android-ndk-6.0.2-android23-arm64/src/tensorflow/lite/tools/make/gen/lib/libtensorflow-lite.a(c_api_internal.o): error adding symbols: File in wrong format
clang++: error: linker command failed with exit code 1 (use -v to see invocation)
```

## Resolved issues
