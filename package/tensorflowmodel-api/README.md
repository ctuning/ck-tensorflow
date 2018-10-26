## Installation

**Attention!!!** This package is intended for Linux only

You should have `ck-tensorflow` repository installed:
```bash
$ cd ~/CK
$ ck pull repo --url=https://github.com/dividiti/ck-tensorflow
```

Also you need install library's package >=1.9 (*-cuda* is prefferable)
You can:
* install library:
  ```bash
  $ ck install ck-tensorflow:package:lib-tensorflow-1.11.0-cuda
  ```
* choose library's version among of installed ones
* choose library's version during of program installation 

**Program installation:**
```bash
$ ck install ck-tensorflow:package:tensorflowmodel-api
```

Also you need:
* One or more compatible dataset(s)
  For example:
  ```bash
  $ ck install ck-tensorflow:package:dataset-kitti-min
  ```
  If you want use COCO-dataset(2014) you should install `ck-caffe` repository:
  ```bash
  $ cd ~/CK
  $ ck pull repo --url=https://github.com/dividiti/ck-caffe
  $ ck install ck-caffe:package:dataset-coco-2014

  ```
  **Attention!!!**
  If you have already installed dataset `dataset-coco-2014`, you should probably renew it:
  ```bash
  $ ck detect ck-caffe:soft:dataset.coco --full_path=/PATH_TO/dataset-coco-2014/val2014/COCO_val2014_000000000042.jpg
  ```
  Or in case of package `dataset-kitti-min` reinstall it:
  ```bash
  $ ck install ck-tensorflow:package:dataset-kitti-min
  ```

* One or more model's package(s) installed
  ```bash
  $ ck install ck-tensorflow:package:tensorflowapimodel-faster-rcnn-resnet101-kitti
  $ ck install ck-tensorflow:package:tensorflowapimodel-ssd-mobilenet-v1-coco
  $ ck install ck-tensorflow:package:tensorflowapimodel-ssd-resnet50-v1-fpn-sbp-640x640-coco
  ```

## Running