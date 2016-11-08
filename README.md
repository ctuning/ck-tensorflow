Collective Knowledge repository for evaluating and optimising performance of TensorFlow
=======================================================================================

# Contributors

* Vladislav Zaborovskiy
* Grigori Fursin, [dividiti](http://dividiti.com) / [cTuning foundation](http://ctuning.org)
* Anton Lokhmotov, [dividiti](http://dividiti.com)

# License
* [BSD](https://github.com/dividiti/ck-caffe/blob/master/LICENSE) (3 clause)

# Status
Under development.

# Installing CK-TensorFlow

## Installing CK on Linux
```
$ sudo pip install ck
```

## Installing CK on Windows
```
$ pip install ck
```

## Obtaining this repository with all dependencies
```
$ ck pull repo:ck-tensorflow
```

## Installing TensorFlow on Ubuntu

## Extra dependencies for GPU version

If you want to use GPU, make sure you have installed CUDA toolkit >=7.0 and cuDNN v2+. If you want to use GPU and pip installation install CUDA toolkit 7.5 and cuDNN v4. 

Check if you have [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus).

[Download and install CUDA](https://developer.nvidia.com/cuda-downloads).

[Download and install cuDNN](https://developer.nvidia.com/rdp/cudnn-download) (Registration required)

[More instructions](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#optional-install-cuda-gpus-on-linux)

### For pip installation

For python 2.7 install
```
$ sudo apt-get install  \
    python-dev \
    python-pip
```

For python 3.5 install 

```
$ sudo apt-get install  \
    python3-dev \
    python3-pip
```

# Installing TensorFlow via CK

We are now ready to install and run CK-TensorFlow:
```
$ ck pull repo:ck-tensorflow
$ ck install package:lib-tensorflow-cpu
```

# Testing installation via image classification

```
 $ ck run program:tensorflow-classification
```

Note, that you will be asked to select a jpeg image from available CK data sets.
We added standard demo images (cat.jpg, catgrey.jpg, fish-bike.jpg, computer_mouse.jpg)
to the ['ctuning-datasets-min' repository](https://github.com/ctuning/ctuning-datasets-min).
You can list them via
```
 $ ck pull repo:ctuning-datasets-min
 $ ck search dataset --tags=dnn
```

# Related projects and initiatives

We are trying to unifying performance analysis and tuning of various DNN frameworks
using Collective Knowledge Technology:
* [CK-Caffe](https://github.com/dividiti/ck-caffe)
* [CK-TinyDNN](https://github.com/ctuning/ck-tiny-dnn)
* [Android app for DNN crowd-benchmarking and crowd-tuning](https://play.google.com/store/apps/details?id=openscience.crowdsource.video.experiments)
* [CK-powered ARM workload automation](https://github.com/ctuning/ck-wa)

# Related Publications with long term vision

* <a href="https://github.com/ctuning/ck/wiki/Publications">All references with BibTex</a>

# Troubleshooting

TensorFlow installation may occasionally fail due to failing to download
some dependencies from GitHub. Restart package installation several times
until Bazel downloads all necessary files.

Sometimes, after installation, TensorFlow crashes with undefined "syntax".
It is usually related to outdated default protobuf (you need version >=3.0.0a4).
To fix this problem upgrade protobuf via
```
 $ pip install protobuf --upgrade
  or
 $ pip3 install protobuf --upgrade
```
