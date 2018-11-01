# TensorFlow object-detection program

## Pre-requisites

### Repositories

```bash
$ ck pull repo:ck-tensorflow
```

You may need `ck-caffe` repository too as it contains packages for object detection datasets:
```bash
$ ck pull repo --url=https://github.com/dividiti/ck-caffe
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

**NB:** If you have already installed dataset `coco` or `kitti`, you should probably renew them:
```bash
$ ck refresh env:{dataset-env-uoa}
```
where `dataset-env-uoa` it one of env identifiers returned by 
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

A number of images t be processed.

Default: `1`

#### `CK_SAVE_IMAGES`

Save processed images with detected boxes painted above.

Default: `NO`

#### `CK_METRIC_TYPE`

A way to calculate metrics.

Available values:
- `KITTI` (default for `dataset-kitti-*`)
- `COCO` (default for `dataset-coco-*`)
- `VOC` (default for `dataset-voc-*`) (**TBD**)
- `OID` (default for `dataset-oid-*`) (**TBD**)

If the parameter is not set then a tool specific for selected dataset will be used.

```bash
$ ck run object-detection-tf-py --env.CK_METRIC_TYPE=KITTI
```
