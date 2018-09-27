# TensorFlow (C++) image classification program

This program uses a statically linked TensorFlow (C++) library.

## Prerequisites

### SciPy

```
# apt install liblapack-dev libatlas-dev
# python -m pip install scipy
```

### TensorFlow library

```
$ ck install package:lib-tensorflow-1.9.0-src-static [--target_os=android23-arm64]
```

**NB:** Use `--target_os=android23-arm64` to build for Android API 23 (v6.0
"Marshmallow") or
[similar](https://source.android.com/setup/start/build-numbers).

### TensorFlow models

Install a TensorFlow model providing a _frozen_ graph via:

```
$ ck install package --tags=tensorflowmodel,frozen
```
or directly via e.g.:
```
$ ck install package:tensorflowmodel-mobilenet-v1-1.0-224-py
```

### ImageNet dataset

```
$ ck install package:imagenet-2012-aux 
$ ck install package --tags=dataset,imagenet,raw,val
```

## Build

```
$ ck compile program:image-classification-tf-cpp [--target_os=android23-arm64]
```

## Run

```
$ ck run program:image-classification-tf-cpp [--target_os=android23-arm64]
```

## Program parameters

### Input image parameters

#### `CK_IMAGE_FILE`

If set, the program will classify a single image instead of iterating over a
dataset.  When only the name of an image is specified, it is assumed that the
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

#### `CK_SUBTRACT_MEAN`

If set to `YES`, then the mean value will be subtracted from the input image.

Default: `YES`.
