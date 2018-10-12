# KITTI evaluation tool

Official KITTI evaluation tool is located [here](http://www.cvlibs.net/datasets/kitti/eval_object.php). But it requires `boost` as additional dependency, so included source is taken from this [SqueezeDet demo](https://github.com/BichenWuUCB/squeezeDet) and it has no dependencies.

The code is slightly modified to be able also taking parameters from environment variables in addition to command line arguments.

## Build
```
ck compile program:kitti-eval-tool
```

## Run
```
ck run program:kitti-eval-tool
```
This program is mainly used internally by various detection programs, e.g. `tensorflow-detection-squeezedet`.

## Environment variables

### `CK_KITTI_LABELS_DIR`
Path to KITTI annotations that will be used for validation. E.g. `${CK-TOOLS}/KITTI-full/training/label_2`

### `CK_IMAGE_LIST_FILE`
Path to a text file containing list of paths to images that were processed by a detection program.

### `CK_RESULTS_DIR`
Path to a directory containig detection results.

### `CK_IMAGE_COUNT`
Number of processed images.

## Detection result file format
TODO
