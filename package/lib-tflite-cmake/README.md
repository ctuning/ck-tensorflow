# TensorFlow Lite package

This package builds TensorFlow Lite (TFLite) using CMake from fixed revisions from master (internally tagged `v2.3.90`, `v2.3.91`, etc).

| Version | Revision |
|-|-|
| `v2.3.90` | 5c1c1085fe331de3 |


## Installation

Build TFLite with one of the supported backends:

```bash
$ ck install package --tags=lib,tflite,v2.3.90,with.eigen
$ ck install package --tags=lib,tflite,v2.3.90,with.ruy
$ ck install package --tags=lib,tflite,v2.3.90,with.xnnpack
$ ck install package --tags=lib,tflite,v2.3.90,with.xnnpack,with.ruy
```

**NB:** Optionally, to restrict the number of CPU cores used to build, append
`--env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=X` to the command (where `X` is the
number of cores to use).
