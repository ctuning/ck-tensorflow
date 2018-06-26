# Classification program for TensorFlow Lite

## Prerequisites

### TensorFlow Lite package

This demo program uses a statically linked TensorFlow Lite library. 

```
$ ck install package:lib-tflite-1.7.0-src-static [--target_os=android23-arm64]
```

**NB:** Use `--target_os=android23-arm64` to build for Android API 23 (v6.0 "Marshmallow") or [similar](https://source.android.com/setup/start/build-numbers).

### Weights package

Install a model providing a graph as tflite file e.g.:

```
$ ck install package:tensorflowmodel-mobilenet-v1-1.0-224-2018_02_22-py
$ ck install package:tensorflowmodel-mobilenet-v2-1.0-224-py 
```


## Build

```
$ ck compile program:tflite-classification [--target_os=android23-arm64]
```

## Run

```
$ ck run program:tflite-classification [--target_os=android23-arm64]
```
