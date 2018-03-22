# Pre-requisites

## Python

### Misc
```
# pip2 install enum34 mock pillow
```

### SciPy

```
# apt install liblapack-dev libatlas-dev
# pip install scipy
```

#### OpenBLAS?
```
$ ck install package:lib-openblas-0.2.20-universal
$ ck search env --tags=openblas,v0.2.20
$ ck load env:<...>
$ cd $INSTALL_DIR/src
$ sudo PREFIX=/usr/local make install 
```

## Install via CK

### TensorFlow

```
$ ck install package:lib-tensorflow-1.4.0-src-{cpu,cuda}{,-xla}
```

### TensorFlow models

```
$ ck install package:tensorflow-model-alexnet-py
$ ck install package:tensorflow-model-squeezenet-py
$ ck install package:tensorflow-model-googlenet-py
```

### ImageNet dataset

```
$ ck install package:imagenet-2012-val-min
$ ck install package:imagenet-2012-aux
```