# TensorFlow program for Object Detection

1. [Setup](#setup)
2. [Usage](#usage)
3. [Adding new models](#add_models)
    - [standardized](#add_model_zoo) (from the [TF model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md))
    - [custom](#add_model_custom)

<a name="setup"></a>
## Setup

### Install CK

Please follow the [CK installation instructions](https://github.com/ctuning/ck#installation).

### Pull CK repositories

```bash
$ ck pull repo:ck-tensorflow
```

### Install TensorFlow

Install TensorFlow v1.14 from source (built with Bazel):
```bash
$ ck install package --tags=lib,tensorflow,v1.14,vsrc,vcpu
```
or from a binary `x86_64` package (installed via `pip`):
```bash
$ ck install package --tags=lib,tensorflow,v1.14,vprebuilt,vcpu
```
Replace `vcpu` with `vcuda` to install TensorFlow with GPU support.

Or you can choose interactively from any available version of TensorFlow:
```bash
$ ck install package --tags=lib,tensorflow
```

### Install TensorFlow models
```bash
$ ck install package --tags=tensorflow,model,api
```

Install one or more object detection model package:
```bash
$ ck install package --tags=model,tf,object-detection --no_tags=deprecated

More than one package or version found:

 0) model-tf-ssd-resnet50-fpn-coco  Version 20180703  (09baac5e6f931db2)
 1) model-tf-faster-rcnn-resnet101-kitti  Version 20170714  (36131254c4390390)
 2) model-tf-yolo-v3-coco  Version reference  (804598aedce689d7)
 3) model-tf-ssdlite-mobilenetv2-kitti  Version reference  (45ed13bcc31c47b6)
 4) model-tf-ssdlite-mobilenetv2-coco  Version reference  (4214ad911ef4c27d)
 5) model-tf-ssd-mobilenetv1-fpn-coco  Version reference  (eddc13966e0464f9)
 6) model-tf-ssd-inceptionv2-coco  Version reference  (b52f64ae9aede4dd)
 7) model-tf-mlperf-ssd-mobilenet-quantized-finetuned  Version finetuned  (9e5de6f4f46b0da0)
 8) model-tf-mlperf-ssd-mobilenet  Version reference  (4134959be0eb9044)
 9) model-tf-faster-rcnn-resnet50-lowproposals-coco  Version reference  (640458144a59763d)
10) model-tf-faster-rcnn-resnet101-lowproposals-coco  Version reference  (38551054ceabf2e2)
11) model-tf-faster-rcnn-nas-lowproposals-kitti  Version reference  (b34f0bb0f9ebe5b9)
12) model-tf-faster-rcnn-nas-lowproposals-coco  Version reference  (7bb51088c44c2b80)
13) model-tf-faster-rcnn-nas-coco  Version reference  (ec32e4a0ead6dfad)
14) model-tf-faster-rcnn-inceptionv2-coco  Version reference  (c3513486696387c2)
15) model-tf-faster-rcnn-inception-resnetv2-atrous-lowproposals-coco  Version reference  (827991e9114dc991)
```

Models with the `coco` suffix are trained on the [COCO dataset](http://cocodataset.org/). See more information on their accuracy and CK packages [here](https://github.com/ctuning/ck-object-detection/tree/master/docker/object-detection-tf-py.tensorrt.ubuntu-18.04#models). Models with the `kitti` suffix are trained or finetuned on the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).
 
 
### Install datasets

#### COCO
```bash
$ ck install package --tags=dataset,object-detection,coco

More than one package or version found:

 0) dataset-coco-2017-val  Version 2017  (04622c746f287473)
 1) dataset-coco-2014-val  Version 2014  (4330ea8a9b47ac90)
```

#### KITTI
```bash
$ ck install package --tags=dataset,object-detection,kitti

More than one package or version found:
 0) KITTI (min)  Version min  (ed443ec82e60b5b5)
 1) KITTI (full)  Version full  (afb43a918fa8758c)
```

**NB:** If you have installed the COCO or KITTI datasets a while ago,
you should probably renew them:
```bash
$ ck refresh env:{dataset-env-uoa}
```
where `dataset-env-uoa` is one of the env identifiers returned by:
```bash
$ ck show env --tags=dataset,kitti
$ ck show env --tags=dataset,coco
```

<a name="usage"></a>
## Usage

### Run the program
```bash
$ ck run program:object-detection-tf-py
```

### Program parameters

#### `CK_CUSTOM_MODEL`

Specifies if the model comes from the TF model zoo or from another source.
A model from somewhere else needs to have some functions reimplemented, as explained [below](#add_model_custom).

Possible values: `0,1`
Default: `0`

#### `CK_ENABLE_BATCH`
Specifies if the image batch processing feature has to be activated or not.

Possible values: `0,1`
Default: `0`

#### `CK_BATCH_SIZE`, `CK_BATCH_COUNT`

The number of images to be processed in a single batch, and the number of batches to process. The total number of images processed is the product of these two parameters.

Possible values: `any positive integer`
Default: `1`

#### `CK_ENABLE_TENSORRT`

Enables the TensorRT backend if the installed TensorFlow library supports it. (Currently, build TensorFlow with CUDA support from sources with the `--env.CK_TF_NEED_TENSORRT="YES"` flag.)

Possible values: `0,1`
Default: `0`

#### `CK_TENSORRT_DYNAMIC`

Enables the [TensorRT dynamic mode](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#static-dynamic-mode).

Possible values: `0,1`
Default: `0`

#### `CK_ENV_IMAGE_WIDTH`, `CK_ENV_IMAGE_HEIGHT`

These parameters can be used to resize at runtime the input images to a different size than the default for the model. This usually decreases the accuracy.
 
Possible values: `any positive integer`
Default:  `Model-specific (set by CK)`

#### `CK_SAVE_IMAGES`

Save processed images with detected boxes overlaid on top.

Possible values: `YES, NO`
Default: `NO`

#### `CK_METRIC_TYPE`

The way to calculate metrics.

Available values:

##### `KITTI` 
Use the evaluation method from the [TF models repository](https://github.com/tensorflow/models/tree/master/research/object_detection/metrics) for Pascal VOC, adapted to the KITTI dataset.

##### `COCO`
Use the evaluation method from the official [MSCOCO evaluation protocol](http://cocodataset.org/#detections-eval) implemented as the CK package `ck-env:package:tool-coco`. Default for `dataset-coco-*`.

##### `COCO_TF`
Use the evaluation method from the [TF models repository](https://github.com/tensorflow/models/tree/master/research/object_detection/metrics) implemented as the CK package `ck-tensorflow:package:tensorflowmodel-api`.

##### `VOC` (**TBD**)
Default for `dataset-voc-*`

##### `OID` (**TBD**)
Default for `dataset-oid-*`

If the parameter is not set, then the tool specific for the selected dataset will be used.

#### `CK_TF_GPU_MEMORY_PERCENT`

Percentage of the GPU memory used by the program

Possible values: `any integer between 1 and 100`
Default: `50`

<a name="add_models"></a>
## Adding new models

The program works with TF models represented as CK packages with tags `model,tf,object-detection`.
Essentially, each package contains JSON metadata and possibly some files.
[Adding a standardized model from the TF model zoo](#add_model_zoo) is straightforward,
while [adding a custom model](#add_model_custom) is a bit more involved.

<a name="add_model_zoo"></a>
### Adding a standardized model

The easiest way to add a new standardized model from the TF zoo is to create a copy of an existing package e.g.:
```
$ ck cp ck-object-detection:package:model-tf-faster-rcnn-nas-coco <repo_name>:package:<model_name>
```
and then update `.cm/meta.json` in the copy.

The most important changes to do are in the `install_env` and `tags` sections
as illustrated below (you need to update anything in angular brackets `<...>`):

```
{
  "check_exit_status": "yes",
  "customize": {
    "extra_dir": "",
    "install_env": {
      "DATASET_TYPE": "coco",
      "DEFAULT_WIDTH": "<your model image width>",
      "DEFAULT_HEIGHT": "<your model image height>",
      "FROZEN_GRAPH": "frozen_inference_graph.pb", 
      "LABELMAP_FILE": "mscoco_label_map.pbtxt",
      "MODEL_NAME": "<your model name>",
      "PACKAGE_NAME": "<your model tarball>.tar.gz",
      "PACKAGE_NAME1": "<your model tarball without extension>",
      "PACKAGE_URL": "<url of the package>",
      "PIPELINE_CONFIG": "pipeline.config",
      "WEIGHTS_FILE": "model.ckpt"
    },
    "model_id": "<model id>",
    "no_os_in_suggested_path": "yes",
    "no_ver_in_suggested_path": "yes",
    "version": "reference"
  },
  "deps": {
    "labelmap": {
      "local": "yes",
      "name": "Labelmap for COCO dataset",
      "tags": "labelmap,vcoco"
    }
  },
  "end_full_path": {
    "linux": "frozen_inference_graph.pb"
  },
  "only_for_host_os_tags": [
    "linux"
  ],
  "only_for_target_os_tags": [
    "linux"
  ],
  "package_extra_name": " (Object Detection <your model name>)",
  "process_script": "install",
  "soft_uoa": "3fc0c4b9ba63de2f",
  "suggested_path": "<your model installation path>",
  "tags": [
    "object-detection",
    "model",
    "tf",
    "tensorflow",
    "<model tags>",
    "<other tags>,
    "vcoco"
  ],
  "use_scripts_from_another_entry": {
    "data_uoa": "c412930408fb8271",
    "module_uoa": "script"
  }
}
```

<a name="add_model_custom"></a>
### Adding a custom model

The easiest way to add a new custom model is to create a copy of an existing custom model package
e.g. [YOLO-v3](https://github.com/ctuning/ck-object-detection/blob/master/package/model-tf-yolo-v3-coco):
```
$ ck cp ck-object-detection:package:model-tf-yolo-v3-coco <repo_name>:package:<model_name>
```
You need to update `.cm/meta.json` in the copy as above. You also need to provide custom
functions in two files called `custom_hooks.py` and `custom_tensorRT.py`. 
Follow the [README](https://github.com/ctuning/ck-object-detection/blob/master/package/model-tf-yolo-v3-coco/README.md) in the example YOLO-v3 package.
