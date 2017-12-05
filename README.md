# Unification of AI for collaborative experimentation and optimization using Collective Knowledge workflow framework with common JSON API

[![logo](https://github.com/ctuning/ck-guide-images/blob/master/logo-powered-by-ck.png)](https://github.com/ctuning/ck)
[![logo](https://github.com/ctuning/ck-guide-images/blob/master/logo-validated-by-the-community-simple.png)](http://cTuning.org)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Introduction

After spending most of our "research" time not on AI innovation but on dealing with numerous 
and ever changing AI engines, their API, and the whole software and hardware stack, 
we decided to take an alternative approach. 

[![logo](http://cknowledge.org/images/ai-cloud-resize.png)](http://cKnowledge.org/ai)

We are developing CK-TensorFlow which is an open-source suite of convenient wrappers and workflows 
powered by [Collective Knowledge](https://github.com/ctuning/ck) with unified JSON API for simple 
and customized installation/recompilation, usage, evaluation and multi-objective optimisation 
of various TensorFlow implementations (CPU,CUDA,OpenCL) across diverse platforms
from mobile devices and IoT to supercomputers and TPU cloud.

See [cKnowledge.org/ai](http://cKnowledge.org/ai), 
[reproducible and CK-powered AI/SW/HW co-design competitions at ACM/IEEE conferences](http://cKnowledge.org/request),
[shared optimization statistics](http://cKnowledge.org/repo),
[reusable AI artifact in the CK format](http://cKnowledge.org/ai-artifacts)
and [online demo of CK AI API with self-optimizing DNN](http://cKnowledge.org/ai/ck-api-demo) for more details.

We need to have easily customizable TensorFlow builds 
via JSON API to be able to plug it to our framework 
for collaborative benchmarking and optimization of realistic
workloads and models (such as deep learning) across diverse inputs 
and hardware provided by volunteers (see [cKnowledge.org/ai](http://cKnowledge.org/ai), 
[live repo](http://cKnowledge.org/repo)
and papers [1](https://arxiv.org/abs/1506.06256), 
[2](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=IwcnpkwAAAAJ&citation_for_view=IwcnpkwAAAAJ:maZDTaKrznsC), 
[3](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=IwcnpkwAAAAJ&citation_for_view=IwcnpkwAAAAJ:LkGwnXOMwfcC) 
for more details).

We can now build TensorFlow library and run classification via CK for various Android and Linux platforms. 
You can even participate in collaborative evaluation and optimization of TF using your Android device
(mobile phone, tablet, etc) via this engaging 
[Android app](https://play.google.com/store/apps/details?id=openscience.crowdsource.video.experiments). 
You can see and use all public results in the [Collective Knowledge repository](http://cKnowledge.org/repo).

# Coordination of development
* [cTuning Foundation](http://cTuning.org)
* [dividiti](http://dividiti.com)

# License
* [BSD](https://github.com/ctuning/ck-tensorflow/blob/master/LICENSE) (3 clause)

# Prerequisites

* Python 2.7+ or 3.3+
* [Collective Knowledge Framework](http://github.com/ctuning/ck)
* Java 8 JDK (though can be automatically installed via CK)
* CUDA/cuDNN if you have [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus)
* Android NDK if you want to compile and run TF for Android devices

## Prerequisites for Ubuntu

### Python

* Python 2.x:
```
$ sudo apt-get install python-dev python-pip python-setuptools python-opencv git
$ sudo pip install --upgrade pip
$ sudo pip install protobuf easydict joblib image numpy scipy
```

* Python 3.x:

```
$ sudo apt-get install python3-dev python3-pip python3-setuptools
$ sudo pip3 install --upgrade pip
$ sudo pip3 install protobuf easydict joblib image wheel numpy scipy
```

## Prerequisites for Windows

```
$ pip install --upgrade pip
$ pip install protobuf easydict joblib image numpy scipy
```

You can find more details about customized TensorFlow builds via CK for Android, Linux, Windows, 
Raspberry Pi, odroid, etc [here](https://github.com/ctuning/ck-tensorflow/wiki/Installation).

### Installing CK and this repository

You can install all basic dependencies and CK as following
```
$ sudo pip install ck
$ ck pull repo:ck-tensorflow
```

## Example of TensorFlow unified installation on Ubuntu or Windows via CK (pre-build binaries)

```
$ ck install package:lib-tensorflow-1.1.0-cpu
$ ck install package:lib-tensorflow-1.1.0-cuda
```

# Benchmarking
```
 $ ck run program:tensorflow (--env.BATCH_SIZE=10) (--env.NUM_BATCHES=5)
```

Select one of the `test_cpu` and `test_cuda` commands;  select an available version of TensorFlow, 
if prompted (more than one choice); select an available benchmark, if prompted (more than one choice),
and select TensorFlow model.

## Example of TensorFlow unified classification on Ubuntu via CK

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
[Android application](https://play.google.com/store/apps/details?id=openscience.crowdsource.experiments) 
soon - please, stay tuned!

## Unified, multi-dimensional and multi-objective autotuning

It is now possible to take advantage of our [universal multi-objective CK autotuner](https://github.com/ctuning/ck/wiki/Autotuning)
to optimize TensorFlow. As a first simple example, we added batch size tuning via CK. You can invoke it as following:

```
$ ck autotune tensorflow
```

All results will be recorded in the local CK repository and 
you will be given command lines to plot graphs or replay experiments such as:
```
$ ck plot graph:{experiment UID}
$ ck replay experiment:{experiment UID} --point={specific optimization point}
```

## Collaborative and unified optimization of DNN

We are now working to extend above autotuner and crowdsource optimization 
of the whole SW/HW/model/data set stack ([paper 1](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=IwcnpkwAAAAJ&citation_for_view=IwcnpkwAAAAJ:maZDTaKrznsC), 
[paper 2](https://arxiv.org/abs/1506.06256)).

We would like to thank the community for their interest and feedback about 
this collaborative AI optimization approach powered by CK 
at [ARM TechCon'16](https://github.com/ctuning/ck/wiki/Demo-ARM-TechCon'16)
and the Embedded Vision Summit'17 - so please stay tuned ;) !

[![logo](http://cKnowledge.org/images/dividiti_arm_stand.jpg)](https://www.researchgate.net/publication/304010295_Collective_Knowledge_Towards_RD_Sustainability)

## Other DNN with unified CK API

CK allows us to unify AI interfaces while collaboratively optimizing underneath engines.
For example, we added similar support to install, use and evaluate [Caffe/Caffe2](https://github.com/ctuning/ck-caffe2) 
and [MXNet](https://github.com/ctuning/ck-mxnet) via CK:

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

## Long term vision

CK-Caffe, CK-Caffe2, CK-Tensorflow are part of an ambitious long-term and community-driven 
project to enable collaborative and systematic optimization 
of realistic workloads across diverse hardware 
in terms of performance, energy usage, accuracy, reliability,
hardware price and other costs
([ARM TechCon'16 talk and demo](https://github.com/ctuning/ck/wiki/Demo-ARM-TechCon'16), 
[DATE'16](http://tinyurl.com/zyupd5v), 
[CPC'15](http://arxiv.org/abs/1506.06256)).

We are working with the community to unify and crowdsource performance analysis 
and tuning of various DNN frameworks (or any realistic workload) 
using Collective Knowledge Technology:
* [Android app for DNN crowd-benchmarking and crowd-tuning](https://play.google.com/store/apps/details?id=openscience.crowdsource.video.experiments)
* [CK-TensorFlow](https://github.com/ctuning/ck-tensorflow)
* [CK-Caffe](https://github.com/dividiti/ck-caffe)
* [CK-Caffe2](https://github.com/ctuning/ck-caffe2)
* [CK-CNTK](https://github.com/ctuning/ck-cntk)
* [CK-TinyDNN](https://github.com/ctuning/ck-tiny-dnn)
* [CK-MVNC (Movidius Neural Compute Stick)](https://github.com/ctuning/ck-mvnc)
* [CK-TensorRT](https://github.com/dividiti/ck-tensorrt)
* [CK-KaNN](https://github.com/dividiti/ck-kann)

We continue gradually exposing various design and optimization
choices including full parameterization of existing models.

## Open R&D challenges

We use crowd-benchmarking and crowd-tuning of such realistic workloads across diverse hardware for 
[open academic and industrial R&D challenges](https://github.com/ctuning/ck/wiki/Research-and-development-challenges.mediawiki) - 
join this community effort!

## Related Publications with long term vision

```
@inproceedings{ck-date16,
    title = {{Collective Knowledge}: towards {R\&D} sustainability},
    author = {Fursin, Grigori and Lokhmotov, Anton and Plowman, Ed},
    booktitle = {Proceedings of the Conference on Design, Automation and Test in Europe (DATE'16)},
    year = {2016},
    month = {March},
    url = {https://www.researchgate.net/publication/304010295_Collective_Knowledge_Towards_RD_Sustainability}
}

@inproceedings{cm:29db2248aba45e59:cd11e3a188574d80,
    url = {http://arxiv.org/abs/1506.06256},
    title = {{Collective Mind, Part II: Towards Performance- and Cost-Aware Software Engineering as a Natural Science.}},
    author = {Fursin, Grigori and Memon, Abdul and Guillon, Christophe and Lokhmotov, Anton},
    booktitle = {{18th International Workshop on Compilers for Parallel Computing (CPC'15)}},
    publisher = {ArXiv},
    year = {2015},
    month = January,
    pdf = {http://arxiv.org/pdf/1506.06256v1}
}

```

* [All references with BibTex related to CK concept](https://github.com/ctuning/ck/wiki/Publications)

## Troublesooting

* SqueezeDet demo currently work well with Python 3.5 and package:squeezedetmodel-squeezedet, so install it first:
```
$ ck install package:squeezedetmodel-squeezedet
$ ck run program:squeezedet
```

## Testimonials and awards

* 2017: We received [CGO test of time award](http://dividiti.blogspot.fr/2017/02/we-received-test-of-time-award-for-our.html) for our CGO'07 paper which later motivated creation of [Collective Knowledge](https://github.com/ctuning/ck)
* 2015: ARM and the cTuning foundation use CK to accelerate computer engineering: [HiPEAC Info'45 page 17](https://www.hipeac.net/assets/public/publications/newsletter/hipeacinfo45.pdf), [ARM TechCon'16 presentation and demo](https://github.com/ctuning/ck/wiki/Demo-ARM-TechCon'16), [public CK repo](https://github.com/ctuning/ck-wa)


## Feedback

Get in touch with CK-AI developers [here](https://github.com/ctuning/ck/wiki/Contacts). Also feel free to engage with our community via this mailing list:
* http://groups.google.com/group/collective-knowledge
