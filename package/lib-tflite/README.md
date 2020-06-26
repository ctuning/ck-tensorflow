# TensorFlow Lite package

This package installs TensorFlow Lite from the following official tagged releases of TensorFlow.

* rel.1.13.1
* rel.1.14.0
* rel.1.15.0 - Enables new GEMM backend of TFLITE
* rel.1.15.3 - 

### Installation

```bash
$ ck install package --tags=lib,tflite,rel.1.15.3 --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=4
```

**NB:** To use machine-specific build options such as for the Raspberry Pi, check
.cm/meta.json to see if the respective variation is available and add it to the installation
command as shown (here in the case of rpi4):


```
$ ck install package --tags=lib,tflite,rel.1.15.3,rpi4
```

or add the relevant extra flags if the relevant hardware
isn't mentioned as shown in the following example


```
$ ck install package --tags=lib,tflite,rel.1.15.3 \
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
