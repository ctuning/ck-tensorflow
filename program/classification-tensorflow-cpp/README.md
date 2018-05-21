# TensorFlow classification demo

## Pre-requisites

### SciPy

```
# apt install liblapack-dev libatlas-dev
# pip2 install scipy
```

### TensorFlow library

This demo uses statically linked TensorFlow library. This package can be installed for Android too.

```
$ ck install package:lib-tensorflow-1.7.0-src-static
$ ck install package:lib-tensorflow-1.7.0-src-static --target_os=android23-arm64
```

### TensorFlow models

Install one of the models providing frozen graph. 

```
$ ck install package:tensorflowmodel-mobilenet-v1-1.0-224-py
```

### ImageNet dataset

```
$ ck install package --tags=dataset,imagenet,raw,val
$ ck install package:imagenet-2012-aux 
```

## Build

```
$ ck complie program:classification-tensorflow-cpp
$ ck complie program:classification-tensorflow-cpp --target_os=android23-arm64
```

## Run

```
$ ck run program:classification-tensorflow-cpp
$ ck run program:classification-tensorflow-cpp --target_os=android23-arm64
```

## Program parameters

### `CK_IMAGE_FILE`
Classify single file instead of iterating over ImageNet dataset.
When only image name is specified, it is assumed that image is in ImageNet dataset.
```
$ ck run program:classification-tensorflow-cpp --env.CK_IMAGE_FILE=/tmp/images/path-to-image.jpg
$ ck run program:classification-tensorflow-cpp --env.CK_IMAGE_FILE=ILSVRC2012_val_00000011.JPEG
```

### `CK_TMP_IMAGE_SIZE`
Preprocessing parameter, size of intermediate image. If this parameter is set to a value greater than targer image size defined by a model, loaded images will be scaled to this size and then cropped to target size.

For example, when running against MobileNet you may specify `--env.CK_TMP_IMAGE_SIZE=256`, then images will be resized to 256x256 the cropped to 224x244 as required to MobileNet.

Default: `0`

### `CK_CROP_PERCENT`
Preprocessing parameter, percentage of central image region for cropping. If this parameter is set to a value between 0 and 100, loaded images will be cropped to this percent and then scaled to targer image size defined by a model.

Not used if `CK_TMP_IMAGE_SIZE` is set and valid.

Default: `87.5`

### `CK_SUBTRACT_MEAN`
Preprocessing parameter, defines whether program should subtract mean value from loaded image.

Default: `YES`

### `CK_RECREATE_CACHE`
Is set to `YES` then existed cached images will be erased. 

Default: `NO`
