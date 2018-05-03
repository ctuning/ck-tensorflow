# Pre-requisites

## Python 2

### Misc
```
# pip2 install enum34 mock pillow
# pip2 install wheel absl-py
```

### SciPy

```
# apt install liblapack-dev libatlas-dev
# pip2 install scipy
```

## Install via CK

### TensorFlow

```
$ ck install package:lib-tensorflow-1.4.0-src-{cpu,cuda}{,-xla}
```

### TensorFlow models

```
$ ck install package:tensorflowmodel-alexnet-py
$ ck install package:tensorflowmodel-squeezenet-py
$ ck install package:tensorflowmodel-googlenet-py
$ ck install package:tensorflowmodel-mobilenet-v1-1.0-224-py
$ ck show env --tags=tensorflowmodel
Env UID:         Target OS: Bits: Name:                                                   Version: Tags:

ef7343498dbec627   linux-64    64 TensorFlow python model and weights (squeezenet)        ImageNet 64bits,host-os-linux-64,python,squeezenet,target-os-linux-64,tensorflow-model,tensorflowmodel,v0,weights
dede2b537d476299   linux-64    64 TensorFlow python model and weights (mobilenet-1.0-224) ImageNet 64bits,host-os-linux-64,mobilenet,mobilenet-v1,mobilenet-v1-1.0-224,python,target-os-linux-64,tensorflow-model,tensorflowmodel,v0,weights
73619b7df1e2488e   linux-64    64 TensorFlow python model and weights (googlenet)         ImageNet 64bits,googlenet,host-os-linux-64,python,target-os-linux-64,tensorflow-model,tensorflowmodel,v0,weights
4dd098ad717db21d   linux-64    64 TensorFlow python model and weights (alexnet)           ImageNet 64bits,alexnet,host-os-linux-64,python,target-os-linux-64,tensorflow-model,tensorflowmodel,v0,weights
```

### ImageNet dataset

```
$ ck install package:imagenet-2012-val-min
$ ck install package:imagenet-2012-aux
```

## Benchmark

```
$ ck list local:experiment:*
$ cd `ck find program:classification-tensorflow`
$ python benchmark.nvidia-gtx1080.py
```

## Program parameters

### `CK_TMP_IMAGE_SIZE`
Preprocessing parameter, size of intermediate image. If this parameter is set to a value greater than targer image size defined by a model, loaded images will be scaled to this size and then cropped to target size.

For example, when running against MobileNet you may specify `--env.CK_TMP_IMAGE_SIZE=256`, then images will be resized to 256x256 the cropped to 224x244 as required to MobileNet.

Default: `0`

### `CK_CROP_PERCENT`
Preprocessing parameter, percentage of central image region for cropping. If this parameter is set to a value between 0 and 100, loaded images will be cropped to this percent and then scaled to targer image size defined by a model.

Not used if `CK_TMP_IMAGE_SIZE` is set and valid.

Default: `87.5`

### `CK_SUBTRACT_MEAN`
Preprocessing parameter, defines whether program should subtract mean value from loaded image. If `CK_USE_MODEL_MEAN` is not set then mean value is calculated over all images' pixels.

Default: `YES`

### `CK_USE_MODEL_MEAN`
Preprocessing parameter, defines whether program should ask a model for mean value that will be subtracted. Model should provide `get_mean_value` method for this.

Used when `CK_SUBTRACT_MEAN` is set.

Default: `YES`

### `CK_CACHE_IMAGES`
Do caching of preprocessed images. Images are cached into a directory whose name contained of preprocessing parameters. Next time when program runs with the same preprocessing parameters, preprocessed images will be loaded from cache. This significantly speeds up images loading process.

Default: `YES`

### `CK_RECREATE_CACHE`
Is set to `YES` then existed cached images will be erased. 

Default: `NO`

### `CK_CACHE_DIR`
Root director for storing cached images. This directory will include additional subdirectories for images preprocessed with different preprocessing parameters `CK_TMP_IMAGE_SIZE` and `CK_CROP_PERCENT`.

Default: `../preprocessed`