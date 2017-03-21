Collective Knowledge repository for evaluating and optimising performance of TensorFlow
=======================================================================================

[![logo](https://github.com/ctuning/ck-guide-images/blob/master/logo-powered-by-ck.png)](http://cKnowledge.org)
[![logo](https://github.com/ctuning/ck-guide-images/blob/master/logo-validated-by-the-community-simple.png)](http://cTuning.org)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Introduction

We need to have easily customizable TensorFlow builds 
via JSON API to be able to plug it to our framework 
for collaborative benchmarking and optimization of workloads 
across diverse inputs and hardware provided by volunteers 
(see [cKnowledge.org/ai](http://cKnowledge.org/ai), 
[live repo](http://cKnowledge.org/repo)
and papers [1](https://arxiv.org/abs/1506.06256), 
[2](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=IwcnpkwAAAAJ&citation_for_view=IwcnpkwAAAAJ:maZDTaKrznsC), 
[3](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=IwcnpkwAAAAJ&citation_for_view=IwcnpkwAAAAJ:LkGwnXOMwfcC) 
for more details).

![](http://cKnowledge.org/images/ai-cloud-resize.png)

# Contributors

* Vladislav Zaborovskiy
* Grigori Fursin, [dividiti](http://dividiti.com) / [cTuning foundation](http://ctuning.org)
* Anton Lokhmotov, [dividiti](http://dividiti.com)

# License
* [BSD](https://github.com/dividiti/ck-caffe/blob/master/LICENSE) (3 clause)

# Status
Under development.

# Installing CK-TensorFlow dependencies

## Installing CK

### Ubuntu
```
$ sudo pip install ck
```

### Windows
```
$ pip install ck
```

## Installing CK-TensorFlow with all its dependencies
```
$ ck pull repo:ck-tensorflow
```

## Installing TensorFlow dependencies on Ubuntu

### Java
```
$ sudo apt install openjdk-8-jdk-headless
```
**NB:** Installation fails with `openjdk-9-jdk-headless` (cf. [this](https://github.com/bazelbuild/bazel/issues/1456)).

### CUDA, cuDNN (GPU version only)

If you want to use the GPU, please install CUDA toolkit >= v7.0 and cuDNN >= v2.
If you want to use the GPU and pip, please install CUDA toolkit >= v8.0 and cuDNN >= v5.

- Check if you have a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus).

- [Download and install CUDA](https://developer.nvidia.com/cuda-downloads).

- [Download and install cuDNN](https://developer.nvidia.com/rdp/cudnn-download) (requires registration).

- [More detailed instructions](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#optional-install-cuda-gpus-on-linux).

### For installation via pip

For python 2.7, install:
```
$ sudo apt-get install  \
    python-dev \
    python-pip \
    python-setuptools

$ sudo pip install wheel
```

For python 3.5, install:

```
$ sudo apt-get install  \
    python3-dev \
    python3-pip \
    python3-setuptools
$ sudo pip3 install wheel
```

### Python
```
$ sudo pip install \
    protobuf \
    easydict \
    joblib
```

# Installing TensorFlow via CK

You should now be ready to install the CPU version of CK-TensorFlow:
```
$ ck install package:lib-tensorflow-cpu
```

You can also install the CUDA version of TensorFlow 
(which when installed via CK can co-exist with the CPU version):
```
$ ck install package:lib-tensorflow-cuda
```

Finally, you can try to install the OpenCL version of TensorFlow
(which requires ComputeCPP; unfortunately, at the time of this writing we were not able to run it):
```
$ ck install package:lib-tensorflow-opencl
```

# Benchmarking
```
 $ ck run program:tensorflow (--env.BATCH_SIZE=10) (--env.NUM_BATCHES=5)
```
Select one of the `benchmark_cpu` and `benchmark_cuda` commands;  select an available version of TensorFlow, if prompted (more than one choice); select an available benchmark, if prompted (more than one choice).  

# Testing (image classification)

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

## Crowd-benchmarking
It is now possible to participate in crowd-benchmarking of Caffe
(early prototype):
```
$ ck crowdbench tensorflow --user={your email or ID to acknowledge contributions} (--env.BATCH_SIZE=128 --env.NUM_BATCHES=100)
```

You can see continuously aggregated results in the 
[public Collective Knowledge repository](http://cknowledge.org/repo)
under 'crowd-benchmark TensorFlow library' scenario.

Note, that this is an on-going, heavily evolving and long-term project
to enable collaborative and systematic benchmarking
and tuning of realistic workloads across diverse hardware 
([ARM TechCon'16 talk](http://schedule.armtechcon.com/session/know-your-workloads-design-more-efficient-systems), 
[ARM TechCon'16 demo](https://github.com/ctuning/ck/wiki/Demo-ARM-TechCon'16), 
[DATE'16](http://tinyurl.com/zyupd5v), [CPC'15](http://arxiv.org/abs/1506.06256)).
We also plan to add crowd-benchmarking and crowd-tuning of Caffe, TensorFlow 
and other DNN frameworks to our 
[Android application](https://play.google.com/store/apps/details?id=openscience.crowdsource.experiments) 
soon - please, stay tuned!

# Troubleshooting

TensorFlow installation may occasionally fail due to failing to download
some dependencies from GitHub. Restart package installation several times
until Bazel downloads all necessary files.

Sometimes, after installation, TensorFlow crashes with undefined "syntax".
It is usually related to outdated default protobuf (you need version >=3.0.0a4).
To fix this problem upgrade protobuf via
```
 $ sudo pip install protobuf --upgrade
  or
 $ sudo pip3 install protobuf --upgrade
```

It may also fail with the following message "can't combine user with prefix, exec_prefix/home, or install_(plat)base".
The following fix may help:
```
 $ sudo pip install --upgrade pip"
```

# Related projects and initiatives

We are working with the community to unify and crowdsource performance analysis 
and tuning of various DNN frameworks (or any realistic workload) 
using Collective Knowledge Technology:
* [CK-Caffe](https://github.com/dividiti/ck-caffe)
* [CK-TinyDNN](https://github.com/ctuning/ck-tiny-dnn)
* [Android app for DNN crowd-benchmarking and crowd-tuning](https://play.google.com/store/apps/details?id=openscience.crowdsource.video.experiments)
* [CK-powered ARM workload automation](https://github.com/ctuning/ck-wa)

# Open R&D challenges

We use crowd-benchmarking and crowd-tuning of such realistic workloads across diverse hardware for 
[open academic and industrial R&D challenges](https://github.com/ctuning/ck/wiki/Research-and-development-challenges.mediawiki) - 
join this community effort!

# Related Publications with long term vision

* <a href="https://github.com/ctuning/ck/wiki/Publications">All references with BibTex</a>
