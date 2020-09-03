# Collective Knowledge components for TensorFlow

*This project is hosted by the [cTuning foundation (non-profit R&D organization)](https://cTuning.org)*

[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![automation](https://github.com/ctuning/ck-guide-images/blob/master/ck-artifact-automated-and-reusable.svg)](http://cTuning.org/ae)
[![workflow](https://github.com/ctuning/ck-guide-images/blob/master/ck-workflow.svg)](http://cKnowledge.org)

[![DOI](https://zenodo.org/badge/65807155.svg)](https://zenodo.org/badge/latestdoi/65807155)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Linux/MacOS: [![Travis Build Status](https://travis-ci.org/ctuning/ck-tensorflow.svg?branch=master)](https://travis-ci.org/ctuning/ck-tensorflow)
Windows: [![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/github/ctuning/ck-tensorflow?branch=master&svg=true)](https://ci.appveyor.com/project/ens-lg4/ck-tensorflow)

# Introduction

CK-TensorFlow repository provides automation components in the [CK format](https://github.com/ctuning/ck) 
for tedious and repetitive tasks such as detecting and installing different TensorFlow versions, models and data sets 
across diverse platforms and running AI/ML workflows in a unified way.

**CK is a collaborative project and not a magic ;)** - if some third-party automation fails 
or misses some functionality (software detection, package installation, bechmarking and autotuning workflow, etc),
the CK concept is to continuously and collaboratively improve such reusable components! 
Please provide your feedback and report bugs via GitHub issues
or get in touch with the community via [public CK mailing list](https://groups.google.com/forum/#!forum/collective-knowledge)!


# Installation

## Prerequisites for Ubuntu

* Python 2.x:
```
$ sudo apt-get install python-dev python-pip python-setuptools python-opencv git
```

or

* Python 3.x:

```
$ sudo apt-get install python3-dev python3-pip python3-setuptools
```

Note that CK will automatically install the following dependencies into CK TF virtual space:
```protobuf easydict joblib image wheel numpy scipy absl-py```

## Optional dependencies depending on your use cases:

* CUDA/cuDNN if you have [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus)

* Android NDK if you want to compile and run TF for Android devices

## CK installation

Follow these [instructions](https://github.com/ctuning/ck#installation) to install CK.

## Installation of ck-tensorflow repository

```
$ ck pull repo:ck-tensorflow
```



# Basic usage

## Example of a unified TensorFlow installation on Ubuntu or Windows via CK (pre-build versions)

```
$ ck install package:lib-tensorflow-1.8.0-cpu
 and/or (CK enables easy co-existance of different versions of tools
$ ck install package:lib-tensorflow-1.8.0-cuda
```

Check that TF is installed locally and registered in the CK:
```
$ ck show env --tags=lib,tensorflow
```

Use CK virtual environment to test it (similar to Python virtual env but for any binary package installed via CK):
```
$ ck virtual env --tags=lib,tensorflow
```

Install other TF versions available in the CK:
```
$ ck install package --tags=lib,tensorflow
```

## Test unified image classification workflow via CK using above TF

```
$ ck run program:tensorflow --cmd_key=classify
```

Note, that you will be asked to select a jpeg image from available CK data sets.
We added standard demo images (cat.jpg, catgrey.jpg, fish-bike.jpg, computer_mouse.jpg)
to the ['ctuning-datasets-min' repository](https://github.com/ctuning/ctuning-datasets-min).
You can list them via
```
 $ ck pull repo:ctuning-datasets-min
 $ ck search dataset --tags=dnn
```

## Customize builds for different platforms

You can find more details about customized TensorFlow builds via CK for Android, Linux, Windows, 
Raspberry Pi, odroid, etc [here](https://github.com/ctuning/ck-tensorflow/wiki/Installation).




# Benchmarking
```
 $ ck run program:tensorflow (--env.BATCH_SIZE=10) (--env.NUM_BATCHES=5)
```

Select one of the `test_cpu` and `test_cuda` commands;  select an available version of TensorFlow, 
if prompted (more than one choice); select an available benchmark, if prompted (more than one choice),
and select TensorFlow model.

## Crowd-benchmarking

It is now possible to participate in crowd-benchmarking of TensorFlow
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
([ARM TechCon'16 talk and demo](https://github.com/ctuning/ck/wiki/Demo-ARM-TechCon'16), 
[DATE'16](http://tinyurl.com/zyupd5v), [CPC'15](http://arxiv.org/abs/1506.06256)).
We also plan to add crowd-benchmarking and crowd-tuning of Caffe, TensorFlow 
and other DNN frameworks to our 
[Android application](http://cKnowledge.org/android-apps.html) 
soon - please, stay tuned!

## Unified, multi-dimensional and multi-objective autotuning

It is now possible to take advantage of our [universal multi-objective CK autotuner](https://github.com/ctuning/ck/wiki/Autotuning)
to optimize TensorFlow. As a first simple example, we added batch size tuning via CK. You can invoke it as follows:

```
$ ck autotune tensorflow
```

All results will be recorded in the local CK repository and 
you will be given command lines to plot graphs or replay experiments such as:
```
$ ck plot graph:{experiment UID}
$ ck replay experiment:{experiment UID} --point={specific optimization point}
```

## Collaborative and unified DNN optimization

We are now working to extend above autotuner and crowdsource optimization 
of the whole SW/HW/model/data set stack ([paper 1](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=IwcnpkwAAAAJ&citation_for_view=IwcnpkwAAAAJ:maZDTaKrznsC), 
[paper 2](https://arxiv.org/abs/1506.06256)).

We would like to thank the community for their interest and feedback about 
this collaborative AI optimization approach powered by CK 
at [ARM TechCon'16](https://github.com/ctuning/ck/wiki/Demo-ARM-TechCon'16)
and the Embedded Vision Summit'17 - so please stay tuned ;) !

[![logo](http://cKnowledge.org/images/dividiti_arm_stand.jpg)](https://www.researchgate.net/publication/304010295_Collective_Knowledge_Towards_RD_Sustainability)

## Using other DNN via unified CK API

CK allows us to unify AI interfaces while collaboratively optimizing underneath engines.
For example, we added similar support to install, use and evaluate [Caffe/Caffe2](https://github.com/ctuning/ck-caffe2),
[CK-PyTorch](https://github.com/ctuning/ck-pytorch) and [MXNet](https://github.com/ctuning/ck-mxnet) via CK:

```
$ ck pull repo:ck-caffe2
$ ck pull repo --url=https://github.com/dividiti/ck-caffe
$ ck pull repo:ck-mxnet

$ ck install package:lib-caffe-bvlc-master-cpu-universal --env.CAFFE_BUILD_PYTHON=ON
$ ck install package:lib-caffe2-master-eigen-cpu-universal --env.CAFFE_BUILD_PYTHON=ON
$ ck install package --tags=mxnet

$ ck run program:caffe --cmd_key=classify
$ ck run program:caffe2 --cmd_key=classify
$ ck run program:mxnet --cmd_key=classify

$ ck crowdbench caffe --env.BATCH_SIZE=5
$ ck crowdbench caffe2 --env.BATCH_SIZE=5 --user=i_want_to_ack_my_contribution

$ ck autotune caffe
$ ck autotune caffe2
```

## Realistic/representative training sets

We provided an option in all our AI crowd-tuning tools to let the community report 
and share mispredictions (images, correct label and wrong misprediction) 
to gradually and collaboratively build realistic data/training sets:
* [Public repository (see "mispredictions and unexpected behavior)](http://cknowledge.org/repo/web.php?action=index&module_uoa=wfe&native_action=show&native_module_uoa=program.optimization)
* [Misclassified images via CK-based AI web-service](http://cknowledge.org/repo/web.php?action=index&module_uoa=wfe&native_action=show&native_module_uoa=program.optimization)

## Online demo of a unified CK-AI API 

* [Simple demo](http://cknowledge.org/repo/web.php?template=ck-ai-basic) to classify images with
continuous optimization of DNN engines underneath, sharing of mispredictions and creation of a community training set;
and to predict compiler optimizations based on program features.

# Open R&D challenges

We use crowd-benchmarking and crowd-tuning of such realistic workloads across diverse hardware for 
[open academic and industrial R&D challenges](https://github.com/ctuning/ck/wiki/Research-and-development-challenges.mediawiki) - 
join this community effort!

# Publications

* [CK publications](https://github.com/ctuning/ck/wiki/Publications)

# Troublesooting

* SqueezeDet demo currently work well with Python 3.5 and package:squeezedetmodel-squeezedet, so install it first:
```
$ ck install package:squeezedetmodel-squeezedet
$ ck run program:squeezedet
```

# Coordination of development
* [cTuning Foundation](http://cTuning.org)
* [dividiti](http://dividiti.com)

# Feedback

Get in touch with ck-tensorflow developers via CK mailing list: http://groups.google.com/group/collective-knowledge !
