# TensorFlow program for Object Detection

## Installation

### Obtain repositories

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


## Run the program
```bash
$ ck run program:object-detection-tf-py
```

### Program parameters

#### `CK_CUSTOM_MODEL`

Specifies if the model comes from the TensorFlow zoo or from another source.
A model from somewhere else needs to have some functions reimplemented, as explained [below](#custom_models).

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
## Add new models

Thanks to the standardization made by Google in the zoo and to the CK framework, it is really easy to integrate a new model from the zoo into the application. 
If you want to add a new model from the zoo, the easiest way to do that it is to copy, using ck, a package containing a model, and to change some parameters inside the .cm/meta.json file.

` ck cp ck-object-detection:package:model-tf-faster-rcnn-nas-coco #your_CK_repo_name#:package:#your_network_name# `


Then inside the meta.json, you will need to change the references like model names, url, tags and so on. The most important changes to do are in the install\_env section and in the tags section, and are reported in the snippet as follows. You will need to replace anything between two hash (#)
`
{
  "check_exit_status": "yes",
  "customize": {
    "extra_dir": "",
    "install_env": {
      "DATASET_TYPE": "coco",
      "DEFAULT_WIDTH": “#your model image width#",
      "DEFAULT_HEIGHT": "#your model image height#",
      "FROZEN_GRAPH": "frozen_inference_graph.pb", 
      "LABELMAP_FILE": "mscoco_label_map.pbtxt",
      "MODEL_NAME": "#your model name#",
      "PACKAGE_NAME": "#your model tarball#.tar.gz",
      "PACKAGE_NAME1": "#your model tarball without extension#",
      "PACKAGE_URL": "#url of the package#",
      "PIPELINE_CONFIG": "pipeline.config",
      "WEIGHTS_FILE": "model.ckpt"
    },
    "model_id": "#model_id#",
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
  "package_extra_name": " #Model Name# ",
  "process_script": "install",
  "soft_uoa": "3fc0c4b9ba63de2f",
  "suggested_path": "#model-installation-path#",
  "tags": [
    "object-detection",
    "model",
    "tf",
    "tensorflow",
    "#model#,
    "#related#,
    "#tags#,
    "vcoco"
  ],
  "use_scripts_from_another_entry": {
    "data_uoa": "c412930408fb8271",
    "module_uoa": "script"
  }
}
`

<a name="custom_models"></a>
### Support for custom models
However, if the model has been created with different input/output tensors, you will have to provide to the application some functions. 
Indeed, the application is structured to work with the models coming from the [tensorflow zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). For this reason, the tensor names and shapes are fixed in a function (`get_handles_to_tensors()`) and also preprocessing and postprocessing are written in order to match the structures for these networks.
However, since we want to be more flexible, we provide a mechanism, through the `CK_CUSTOM_MODEL` parameter, to have the common tensorflow backend working also with models that are not coming from the official TF zoo.
Indeed, the kernel of the detection is structured to have some function calls (defined in the `func_defs` dictionary). These function to call are selected in the `init` function, according to the setup of the application. In particular this function uses three parameters, `CK_ENABLE_BATCH`, `CK_CUSTOM_MODEL` and `CK_ENABLE_TENSORRT` to associate the function call with the implementation of that call.

If someone wish to add a model that has a different structure from the tensorflow zoo, it is possible to do so by writing the appropriate functions.
These function MUST be included in the ck package that is created to support the model, in two files with names `custom_hooks.py` and `custom_tensorRT.py`. the names and parameters have to be coherent with the names and parameters in the program.
An example of these function is provided in the [`yolo`](https://github.com/ctuning/ck-object-detection/tree/master/package/yolo-v3) package.
To add a new custom model, we suggest to start again from an already existing package (in this case the yolo-v3) to have the infrastructure ready.

` ck cp ck-object-detection:package:model-tf-yolo-v3-coco #your_CK_repo_name#:package:#your_network_name# `

After the copy, you should edit the meta.json file as reported above.

Once this is done, the user can follow the README in the yolo package, to see the interface of the required functions that he will have to implement.
