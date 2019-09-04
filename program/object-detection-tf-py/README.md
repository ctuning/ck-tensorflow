# TensorFlow object-detection program

## Pre-requisites

### Repositories

```bash
$ ck pull repo:ck-tensorflow
```

### TensorFlow

Install from source:
```bash
$ ck install package:lib-tensorflow-1.14.0-src-{cpu,cuda}
```
or from a binary `x86_64` package:
```bash
$ ck install package:lib-tensorflow-1.14.0-{cpu,cuda}
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
$ ck install package --tags=tensorflow,model,object-detection

 0) tensorflowmodel-object-detection-ssd-resnet50-v1-fpn-sbp-640x640-coco  Version 20180703  (09baac5e6f931db2)
 1) tensorflowmodel-object-detection-faster-rcnn-resnet101-kitti  Version 20170714  (36131254c4390390)
 2) yolo-v3  Version reference  (804598aedce689d7)
 3) ssdlite-mobilenet-v2-kitti  Version reference  (45ed13bcc31c47b6)
 4) ssdlite-mobilenet-v2  Version reference  (4214ad911ef4c27d)
 5) ssd-mobilenetv1-fpn-shared-box-predictor  Version reference  (eddc13966e0464f9)
 6) ssd-inception-v2  Version reference  (b52f64ae9aede4dd)
 7) model-tf-mlperf-ssd-mobilenet-quantized-finetuned  Version finetuned  (9e5de6f4f46b0da0)
 8) model-tf-mlperf-ssd-mobilenet  Version reference  (4134959be0eb9044)
 9) faster-rcnn_inception-resnet_v2-atrous-lowproposal  Version reference  (827991e9114dc991)
10) faster-rcnn-resnet50-lowproposal  Version reference  (640458144a59763d)
11) faster-rcnn-resnet101-lowproposal  Version reference  (38551054ceabf2e2)
12) faster-rcnn-nas-lowproposal-kitti  Version reference  (b34f0bb0f9ebe5b9)
13) faster-rcnn-nas-lowproposal  Version reference  (7bb51088c44c2b80)
14) faster-rcnn-nas  Version reference  (ec32e4a0ead6dfad)
15) faster-rcnn-inception-v2  Version reference  (c3513486696387c2)

```
Available models characteristics


| Model | Unique CK Tags (`<tags>`) | Is Custom? | Dataset |
| --- | --- | --- | --- |
| [faster\_rcnn\_resnet50\_lowproposals\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)  | `rcnn,lowproposal,resnet50`  | 0 | coco |
| [faster\_rcnn\_resnet101\_lowproposals\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) | `rcnn,lowproposal,resnet101` | 0 | coco |
| [faster\_rcnn\_nas\_lowproposals\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)       | `rcnn,lowproposal,nas,vcoco` | 0 | coco |
| [faster\_rcnn\_nas\_lowproposals\_kitti](TBD)       | `rcnn,lowproposal,nas,vkitti`      | 0 | kitti |
| [faster\_rcnn\_inception\_resnet\_v2\_atrous\_lowproposals\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) | `rcnn,lowproposal,inception,resnetv2` | 0 | coco |
| [faster\_rcnn\_inception\_v2\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)           | `rcnn,inceptionv2`           | 0 | coco |
| [ssd\_mobilenet\_v1\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)            | `ssd-mobilenet,non-quantized,mlperf` | 0 | coco |
| [ssd\_mobilenet\_v1\_quantized\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) | `ssd-mobilenet,quantized`            | 0 | coco |
| [ssd\_mobilenet\_v1\_fpn\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)       | `ssd,fpn`                            | 0 | coco |
| [ssd\_resnet\_50\_fpn\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)          | `ssd,resnet50`                       | 0 | coco |
| [ssd\_inception\_v2\_coco ](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)           | `ssd,inceptionv2`                    | 0 | coco |
| [ssdlite\_mobilenet\_v2\_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)        | `ssdlite,vcoco`                      | 0 | coco |
| [ssdlite\_mobilenet\_v2\_coco](TBD)        | `ssdlite,vkitti`                     | 0 | kitti |
| [yolo\_v3\_coco](https://github.com/YunYang1994/tensorflow-yolov3)                                                                             | `yolo`                               | 1 | coco |
| [faster\_rcnn\_resnet101](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) | `resnet101,vkitti` | 0 | kitti |
 
 



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

#### `CK_CUSTOM_MODEL`

Specifies if the model comes from the TensorFlow zoo or from another source.
Model coming from other sources need to have some functions reimplemented. More on this is explained in [Support for custom models](#custom_models) section.

Possible values: `0,1`
Default: `0`
#### `CK_ENABLE_BATCH`
Specifies if the image batch processing feature has to be activated or not.

Possible values: `0,1`
Default: `0`
#### `CK_BATCH_COUNT`, `CK_BATCH_COUNT`

The number of images to be processed in a single batch, and the number of batches to process. The total number of images processed is the product of these two parameters.

Possible values: `any positive integer`
Default: `1`

#### `CK_ENABLE_TENSORRT`

Enables the TensorRT backend (if the installed tensorflow library supports it. Up to now, only the source installation, with CUDA support, with the flag `--env.CK_TF_NEED_TENSORRT="YES"` is supporting this feature).

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
Use the evaluation method from  [TF models repository](https://github.com/tensorflow/models/tree/master/research/object_detection/metrics) for Pascal VOC, adapted to the KITTI dataset.

##### `COCO`
Use the evaluation method from the official [MSCOCO evaluation protocol](http://cocodataset.org/#detections-eval) implemented as the CK package `ck-env:package:tool-coco`. Default for `dataset-coco-*`.

##### `COCO_TF`
Use the evaluation method from [TF models repository](https://github.com/tensorflow/models/tree/master/research/object_detection/metrics) implemented as the CK package `ck-tensorflow:package:tensorflowmodel-api`.

##### `VOC` (**TBD**)
Default for `dataset-voc-*`

##### `OID` (**TBD**)
Default for `dataset-oid-*`

If the parameter is not set, then the tool specific for the selected dataset will be used.

#### `CK_TF_GPU_MEMORY_PERCENT`

Percentage of the GPU memory used by the program

Possible values: `any integer between 1 and 100`
Default: `33`

<a name="custom_models"></a>
## Support for custom models

The application is structured to work with the models coming from the [tensorflow zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). For this reason, the tensor names and shapes are fixed in a function (`get_handles_to_tensors()`) and also preprocessing and postprocessing are written in order to match the structures for these networks.
However, we want to be more flexible, and we provide a mechanism, through the `CK_CUSTOM_MODEL` parameter, to have the common tensorflow backend working also with models that are not coming from the official TF zoo.
Indeed, the kernel of the detection is structured to have some function calls (defined in the `func_defs` dictionary). These function to call are selected in the `init` function, according to the setup of the application. In particular this function uses three parameters, `CK_ENABLE_BATCH`, `CK_CUSTOM_MODEL` and `CK_ENABLE_TENSORRT` to associate the function call with the implementation of that call.

If someone wish to add a model that has a different structure from the tensorflow zoo, it is possible to do so by writing the appropriate functions.
These function MUST be included in the ck package that is created to support the model, in two files with names `custom_hooks.py` and `custom_tensorRT.py`. the names and parameters have to be coherent with the names and parameters in the program. An example of these function is provided in the `yolo` package.





