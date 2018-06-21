# Common scripts for benchmarking programs

Common preprocessing and postprocessing scripts to be used in benchmarking programs such as `image-classification-tf-py`, `image-classification-tf-cpp`, `image-classification-tflite`, `mobilenets-armcl-opencl` (*TBD*).

A client program has to reference scripts in its meta in the section `run_time`, e.g.:

```json
  "run_cmds": {
    "default": {
      "run_time": {
        "post_process_via_ck": "yes",
        "post_process_cmds": [
          "python $#ck_take_from_{script:689867d1939a781d}#$postprocess.py"
        ],
        "pre_process_via_ck": {
          "module_uoa": "script",
          "data_uoa": "689867d1939a781d",
          "script_name": "preprocess"
        },
        "run_cmd_main": "$#BIN_FILE#$",
```

It is supposed that the client program provides required enviroment variables and dependencies with suitable names that scripts will search for.


## Preprocessing

Preprocessing script prepares images for client programs.

Preprocessing steps:

- Read required number of images from a dataset. The number of images is governed by program parameters `CK_BATCH_COUNT` and `CK_BATCH_SIZE`.
  
- Prepare images for loading into a network. Preparation includes cropping images, scaling them to a size defined by input images size of a network being benchmarked. See the section **Input preprocessing parameters** below.

- Store prepared images into a cache directory.

As a result, preprocessing script provides a set of enviroment variables that client program should use:

- `RUN_OPT_IMAGE_DIR`
Path to a directory containing preprocessed images.
  
- `RUN_OPT_IMAGE_LIST`
Path to a file containing list of images to be processed.
This file contains only image file names (one per line) without paths.

- `RUN_OPT_RESULT_DIR`
Path to a directory to which the client program should store prediction results that will be validated by postprocessing script.

### Program-specific preprocessing

Client program can contain its own additional preprocessing script  `preprocess-next.py`. If it is presented it will be called after the common preprocessing.

## Images dataset

Client program should provide access to the ImageNet dataset via run-time dependencies `imagenet-val` and `imagenet-aux`, e.g.:

```json
  "run_deps": {
    "imagenet-aux": {
      "force_target_as_host": "yes",
      "local": "yes",
      "name": "ImageNet dataset (aux)",
      "sort": 10,
      "tags": "dataset,imagenet,aux"
    },
    "imagenet-val": {
      "force_target_as_host": "yes",
      "local": "yes",
      "name": "ImageNet dataset (val)",
      "sort": 20,
      "tags": "dataset,imagenet,raw,val"
    },
```

## Weights package

A network for benchmarking is provided by program's run-time dependency `weights`, e.g.:

```json
  "run_deps": {
    "weights": {
      "force_target_as_host": "yes",
      "local": "yes",
      "name": "TensorFlow-Python model and weights",
      "no_tags": "mobilenet-all",
      "sort": 30,
      "tags": "tensorflowmodel,weights,tflite"
    }
```

**TODO** Currently only TensorFlow packages provide env variable giving their wanted input image size (`CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH`). But all the packages should do.

## Program parameters

Here are set of client program parameters affecting pre/post-processig stages.

### Input image parameters

#### `CK_IMAGE_FILE`

If set, the program will classify a single image instead of iterating over a
dataset. When only the name of an image is specified, it is assumed that the
image is in the ImageNet dataset.

```
$ ck run program:image-classification-tf-cpp --env.CK_IMAGE_FILE=/tmp/images/path-to-image.jpg
$ ck run program:image-classification-tf-cpp --env.CK_IMAGE_FILE=ILSVRC2012_val_00000011.JPEG
```

#### `CK_RECREATE_CACHE`
If set to `YES`, then all previously cached images will be erased.

Default: `NO`.

### Input preprocessing parameters

#### `CK_TMP_IMAGE_SIZE`

The size of an intermediate image. If this preprocessing parameter is set to a
value greater than the input image size defined by the model, input images
will be first scaled to this size and then cropped to the input size.

For example, if `--env.CK_TMP_IMAGE_SIZE=256` is specified for MobileNets
models with the input image resolution of 224, then input images will be first
resized to *256x256* and then cropped to *224x224*.

Default: `0`.

#### `CK_CROP_PERCENT`

The percentage of the central image region to crop. If this preprocessing
parameter is set to a value between 0 and 100, then loaded images will be
cropped according this percentage and then scaled to the input image size
defined by the model.

Default: `87.5`.

**NB:** If `CK_TMP_IMAGE_SIZE` is set and valid, this parameter is not used.

