# TensorFlow object-detection program

## Pre-requisites

### Repositories

```bash
$ ck pull repo:ck-tensorflow
```

### TensorFlow

Install from source:
```bash
$ ck install package:lib-tensorflow-1.10.1-src-{cpu,cuda}
```
or from a binary `x86_64` package:
```bash
$ ck install package:lib-tensorflow-1.10.1-{cpu,cuda}
```

Or you can choose from different available version of TensorFlow packages:
```bash
$ ck install package --tags=lib,tensorflow
```

### TensorFlow models
```bash
$ ck install ck-tensorflow:package:tensorflowmodel-api
```

Install one or more object detection model package:
```bash
$ ck install package --tags=tensorflowmodel,object-detection

 0) tensorflowmodel-object-detection-ssd-resnet50-v1-fpn-sbp-640x640-coco  Version 20170714  (09baac5e6f931db2)
 1) tensorflowmodel-object-detection-ssd-mobilenet-v1-coco  Version 20170714  (385831f88e61be8c)
 2) tensorflowmodel-object-detection-faster-rcnn-resnet101-kitti  Version 20170714  (36131254c4390390)
```

### Datasets
```bash
$ ck install package --tags=dataset,object-detection
```

**NB:** If you have previously installed the `coco` or `kitti` datasets, you should probably renew them:
```bash
$ ck refresh env:{dataset-env-uoa}
```
where `dataset-env-uoa` is one of the env identifiers returned by:
```bash
$ ck show env --tags=dataset,kitti
$ ck show env --tags=dataset,coco
```

## Running

```bash
$ ck run object-detection-tf-py
```

### Program parameters

#### `CK_BATCH_COUNT`

The number of images to be processed.

Default: `1`

#### `CK_SAVE_IMAGES`

Save processed images with detected boxes overlaid on top.

Default: `NO`

#### `CK_METRIC_TYPE`

The way to calculate metrics.

Available values:

##### `KITTI` (**TBD**)
Default for `dataset-kitti-*`

##### `COCO`
Use the evaluation method from the official [MSCOCO evaluation protocol](http://cocodataset.org/#detections-eval) implemented as the CK package `ck-env:package:tool-coco`. Default for `dataset-coco-*`.

##### `COCO_TF`
Use the evaluation method from [TF models repository](https://github.com/tensorflow/models/tree/master/research/object_detection/metrics) implemented as the CK package `ck-tensorflow:package:tensorflowmodel-api`.

##### `VOC` (**TBD**)
Default for `dataset-voc-*`

##### `OID` (**TBD**)
Default for `dataset-oid-*`

If the parameter is not set, then the tool specific for the selected dataset will be used.

```bash
$ ck run object-detection-tf-py --env.CK_METRIC_TYPE=KITTI
```
