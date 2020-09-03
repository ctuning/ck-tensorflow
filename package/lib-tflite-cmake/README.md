# TensorFlow Lite package

This package builds TensorFlow Lite from the following official tagged releases of TensorFlow:

* `rel.1.13.1`
* `rel.1.14.0`
* `rel.1.15.0` (enables [ruy](https://github.com/google/ruy), a new GEMM backend)
* `rel.1.15.3`
* `rel.2.0.2` ([fails](#build_failure_2_0) on 32-bit Raspberry Pi OS)
* `rel.2.1.0`
* `rel.2.1.1`

## Installation

To install a particular release, use one of the above variation tags e.g.:
```bash
$ ck install package --tags=lib,tflite,rel.1.15.3
```

### Target-specific variations

#### Raspberry Pi 4, with 32-bit [Raspberry Pi OS](https://www.raspberrypi.org/downloads/raspberry-pi-os/)

To use the optimal compilation flags, add the `rpi4` variation tag e.g.:
```bash
$ ck install package --tags=lib,tflite,rel.1.15.3,rpi4
```

**NB:** This is equivalent to:
```bash
$ ck install package --tags=lib,tflite,rel.1.15.3 \
--env.EXTRA_CXXFLAGS="-march=armv7-a+neon+vfpv4 -mfpu=neon-vfpv4"
```

### Build-thread variations

To limit the number of build threads on a resource-constrained platform, add e.g.:
```bash
$ ck install package --tags=lib,tflite,rel.1.15.3,threads.2
```

**NB:** This is equivalent to:
```bash
$ ck install package --tags=lib,tflite,rel.1.15.3 \
--env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=2
```
However, thread variations currently only support 1-4 threads, while `CK_HOST_CPU_NUMBER_OF_PROCESSORS` can be set to higher numbers.

## Known issues

<a name="build_failure_2_0"></a>
### Cannot build v2.0.x on 32-bit Raspberry Pi OS

Building `rel.2.0.2` on a Raspberry Pi 4 with 32-bit Raspberry Pi OS (GCC 8.3.0), runs into an assemler problem:
```
g++ -O3 -DNDEBUG -fPIC -flax-vector-conversions -fomit-frame-pointer -DTFLITE_WITH_RUY -march=armv7-a+neon+vfpv4 -mfpu=neon-vfpv4  --std=c++11 -fPIC -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -pthread -I. -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/../../../../../ -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/../../../../../../ -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/downloads/ -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/downloads/eigen -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/downloads/absl -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/downloads/gemmlowp -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/downloads/neon_2_sse -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/downloads/farmhash/src -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/downloads/flatbuffers/include -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/downloads/googletest/googletest/include/ -I/home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/downloads/googletest/googlemock/include/ -I -I/usr/local/include -c tensorflow/lite/experimental/ruy/thread_pool.cc -o /home/lahiru/CK-TOOLS/lib-tflite-src-static-2.0.2-gcc-8.3.0-rel.2.0.2-rpi4-linux-32/src/tensorflow/lite/tools/make/gen/obj/tensorflow/lite/experimental/ruy/thread_pool.o^M
/tmp/ccvOXh3e.s: Assembler messages:
/tmp/ccvOXh3e.s:114: Error: immediate expression requires a # prefix -- `mov r0,0'
```
It is fine on a Raspberry Pi 4 with 64-bit Ubuntu 20.04 (GCC 9.3.0).

### Cannot build for Android (taken from [`package:lib-tflite-1.15.0-src-static`](https://github.com/ctuning/ck-tensorflow/blob/master/package/lib-tflite-1.15.0-src-static/README.md))

Tried installing with `--target_os=android23-arm64` using Android NDK 17.2:
- GCC 4.9.x complained about paths to `stdint.h`.
- LLVM 6.0.2 failed when linking:
```
/home/anton/CK_TOOLS/lib-tflite-src-static-1.15.0-rc2-llvm-android-ndk-6.0.2-android23-arm64/src/tensorflow/lite/tools/make/gen/lib/libtensorflow-lite.a(c_api_internal.o): error adding symbols: File in wrong format
clang++: error: linker command failed with exit code 1 (use -v to see invocation)
```
