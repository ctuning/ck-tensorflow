# SqueezeDet validation and finetuning demo

This demo is mainly based on another demo by [BichenWuUCB et al.](https://github.com/BichenWuUCB/squeezeDet) and uses it as dependency.

## Requirements

- Tensorflow library:
```
ck install package --tags=lib,tensorflow --no_tags=vshared
```

- KITTI images dataset:
```
ck install package --tags=dataset,kitti
```

- One of SqueezeDet models:
```
ck install package:squeezedetmodel-squeezedet
ck install package:squeezedetmodel-squeezedet-plus
ck install package:squeezedetmodel-resnet50
ck install package:squeezedetmodel-vgg16
```

- SqueezeDet demo:
```
ck install package:demo-squeezedet-patched
```
This is only required as it contains Python modules implementing models.

**TODO:** Including Python modules into respective package listed above we could avoid this dependency.

- KITTI evaluation tool:
```
ck compile program:kitti-eval-tool
```

## Validation

```
ck run program:tensorflow-detection-squeezedet --cmd_key=test
```

## Finetuning (TODO)

```
ck run program:tensorflow-detection-squeezedet --cmd_key=finetune
```

We could implement a command to fine-tune selected SqueezeDet model to detect objects of VOC and COCO datasets.

There are some challenges here:

- Annotations format is different so some kind of label conversion is required.
