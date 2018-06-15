# Test program for TensorFlow Lite ck package.

This program in ck format based on the original TensorFlow Lite example `label_image` (`tensorflow/contrib/lite/examples/label_image/label_image/label_image.cc`).

## Requirements

```bash
ck install package:lib-tflite-1.7.0-src-static
ck install package:lib-tflite-1.7.0-src-staticc --target_os=android23-arm64
```

Model and labels are downloaded from [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip).

## Build

```bash
ck compile program:ch-test-tflite
ck compile program:ch-test-tflite --target_os=android23-arm64
```

## Run

```bash
ck run program:ch-test-tflite
ck run program:ch-test-tflite --target_os=android23-arm64
```
