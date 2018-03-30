# Pre-requisites

## Python 2

### Misc
```
# pip2 install enum34 mock pillow
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
```

### ImageNet dataset

```
$ ck install package:imagenet-2012-val-min
$ ck install package:imagenet-2012-aux
```
