# TensorFlow Lite (C++) object detection program

This program uses a statically linked TensorFlow Lite (C++) library and TensorFlow Lite SSD MobileNet models.

**Note:** It uses TensorFlow Lite [realisation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/detection_postprocess.cc)
of `Non Max Suppresion` for [custom operator](https://www.tensorflow.org/lite/guide/ops_custom).

## Prerequisites

### Repositories

```bash
$ ck pull repo:ck-tensorflow
$ ck pull repo:ck-mlperf
```

### TensorFlow Lite library

```bash
$ ck install package:lib-tflite-1.13.1-src-static
```

### Flatbuffers library

```bash
$ ck install package --tags=lib,flatbuffers
```

### xOpenme library

```bash
$ ck install package --tags=lib,xopenme
```

### TensorFlow models API
```bash
$ ck install ck-tensorflow:package:tensorflowmodel-api
```
or 
```bash
$ ck install package --tags=tensorflowmodel,api
```

### Python libraries
Numpy
```bash
$ ck install package --tags=lib,python-package,numpy
```

Pillow
```bash
$ ck install package --tags=lib,python-package,pillow
```

Matplotlib
```bash
$ ck install package --tags=lib,python-package,matplotlib
```

Python API for COCO-dataset
```bash
$ ck install package --tags=tool,coco
```

### TensorFlow models

Install a TensorFlow model SSD MobileNet via:

```bash
$ ck install ck-tensorflow:package:-object-detection-ssd-mobilenet-v1-coco
```
or:
```
$ ck install ck-mlperf:package:model-tflite-mlperf-ssd-mobilenet
```

### Datasets
```bash
$ ck install package --tags=dataset,object-detection,coco
```

**NB:** If you have previously installed the `coco` datasets, you should probably renew them:
```bash
$ ck refresh env:{dataset-env-uoa}
```
where `dataset-env-uoa` is one of the env identifiers returned by:
```bash
$ ck show env --tags=dataset,coco
```

## Compiling

```bash
$ ck compile ck-tensorflow:program:object-detection-tflite
```

## Running

```bash
$ ck run ck-tensorflow:program:object-detection-tflite
```

### Program parameters

#### `CK_BATCH_COUNT`

The number of images to be processed.

Default: `1`

```bash
$ ck run ck-tensorflow:program:object-detection-tflite --env.CK_BATCH_COUNT=100
```

#### `CK_HOST_CPU_NUMBER_OF_PROCESSORS`

The number of threads used by TF Lite library for job.

Default: `1`

```bash
$ ck run ck-tensorflow:program:object-detection-tflite --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=2
```

#### `FULL_REPORT` or `VERBOSE`

Show additional info (FULL_REPORT) or complete additional info (VERBOSE)

Default: `no`

```bash
$ ck run ck-tensorflow:program:object-detection-tflite --env.FULL_REPORT=yes
```

or:
```bash
$ ck run ck-tensorflow:program:object-detection-tflite --env.VERBOSE=yes
```

#### `USE_CUSTOM_NMS_SETTINGS`

Enable tuning model settings (just for custom operator use).

**Note:** Are you sure you know to do?

Default: `no`

Available settings are:
```
MAX_CLASSES_PER_DETECTION (integer, >0)
MAX_DETECTIONS (integer, >0)
DETECTIONS_PER_CLASS (integer, >0)

NMS_SCORE_THRESHOLD (float, >=0.0)
NMS_IOU_THRESHOLD (float, >=0.0)
SCALE_H (float, >0.0)
SCALE_W (float, >0.0)
SCALE_X (float, >0.0)
SCALE_Y (float, >0.0)
```

Usage example:
```bash
$ ck run ck-tensorflow:program:object-detection-tflite \
    --env.CK_BATCH_COUNT=10 \
    --env.FULL_REPORT=yes \
    --env.USE_CUSTOM_NMS_SETTINGS=yes \
    --env.MAX_DETECTIONS=100 \
    --env.NMS_SCORE_THRESHOLD=0.25 \
    --env.NMS_IOU_THRESHOLD=0.7
```
