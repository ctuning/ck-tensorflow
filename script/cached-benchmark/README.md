# Common scripts for benchmarking programs

These are common preprocessing and postprocessing scripts to be used in benchmarking programs such as `classification-tensorflow` (TBD), `classification-tensorflow-cpp` (TBD), `classification-tflite-cpp`.

**NOTE:** Even through these scripts are currently used by tensorflow programs, they are not directly related to TensorFlow and could be used with other programs, e.g. `mobilenets-armcl-opencl`. So they can be moved to some more common location (where?).

A client program has to reference preprocessing script in its meta in `run_time` section, e.g.:

```json
  "run_cmds": {
    "default": {
      "run_time": {
        "post_process_via_ck": "yes",
        "post_process_cmds": [
          "python $#src_path_local#$postprocess.py"
        ],
        "pre_process_via_ck": {
          "module_uoa": "script",
          "data_uoa": "689867d1939a781d",
          "script_name": "preprocess"
        },
        "run_cmd_main": "$#BIN_FILE#$",
```

It is supposed that client programs provide required enviroment variables and dependencies with suitable names that scripts will search for.


## Preprocessing

Preprocessing script prepares images for client programs.

Preprocessing steps:

- Read required number of images from a dataset.

As a result preprocessing script provides a set of enviroment variables that client program should use.

- `RUN_OPT_IMAGE_DIR`
Path to a directory containing preprocessed images.
  
- `RUN_OPT_IMAGE_LIST`
Path to a file containing list of images to be processed.
This file contains only image file names (one per line) without paths.


### Images dataset

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

 includes cropping images, scaling them to a size defined by input images size of a network being benchmarked.

<!--
## Program parameters

### Input image parameters

#### `CK_IMAGE_FILE`

If set, the program will classify a single image instead of iterating over a
dataset. When only the name of an image is specified, it is assumed that the
image is in the ImageNet dataset.

```
$ ck run program:classification-tensorflow-cpp --env.CK_IMAGE_FILE=/tmp/images/path-to-image.jpg
$ ck run program:classification-tensorflow-cpp --env.CK_IMAGE_FILE=ILSVRC2012_val_00000011.JPEG
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

#### `CK_SUBTRACT_MEAN`

If set to `YES`, then the mean value will be subtracted from the input image.

Default: `YES`.

->