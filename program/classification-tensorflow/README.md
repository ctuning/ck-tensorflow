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
