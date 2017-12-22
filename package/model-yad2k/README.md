# YAD2K: Yet Another Darknet 2 Keras

CK package for [YAD2K](https://github.com/allanzelener/YAD2K).

# Install dependencies

## Install via `apt`

```
# apt install \
  python3     \
  python3-dev \
  python3-pip \
  python3-tk  \
  wget
```

## Install via `pip`

```
# python3 -m pip install \
  numpy                  \
  keras                  \
  matplotlib             \
  h5py                   \
  pillow                 \
  wheel
```

**NB:** The dependencies are needed for:

- `h5py`: serializing Keras model;
- `pillow`: visualizing results;
- `pydot-ng`: plotting model (optional).


## Install via `ck`

### Install `ck`, `ck-env`, `ck-tensorflow`

```
# pip install ck
$ ck pull repo:ck-env
$ ck pull repo:ck-tensorflow
```

### Detect GCC, CUDA

**NB:** Use (CUDA 8 and GCC 5) or (CUDA 9 and GCC 6).
```
$ ck detect soft.compiler.gcc
$ ck detect soft:compiler.cuda
```

### Detect Python, Keras

**NB:** Use Python 3.
```
$ ck detect soft:compiler.python
$ ck detect soft:lib.keras
```

### TensorFlow [`x86_64`]

**NB:** Use Python 3, (CUDA 8 and GCC 5), cuDNN 6.

```
$ ck install package:lib-tensorflow-1.4.0-cuda
```

### TensorFlow [build from sources]

**NB:** Use Java 1.8, Bazel 0.8, Python 3, (CUDA 8 and GCC 5) or (CUDA 9 and GCC 6), cuDNN 7.

```
$ ck install package:jdk-8u131-universal
$ ck install package:tool-bazel-0.8.1-linux
$ ck install package:lib-tensorflow-1.4.0-src-cuda
```

### YAD2K

```
$ ck install package:model-yad2k
$ ck run program:yad2k-demo --cmd_key=convert
```

**FIXME:** The model is currently cloned into `${INSTALL_DIR}`, which needs to be manually deleted if re-installation is attempted. A better approach would be to clone into `${INSTALL_DIR}/src`.

# Run YAD2K demo

```
$ ck run program:yad2k-demo --cmd_key=test
...
Using TensorFlow backend.
Creating output path images/out
2017-12-22 09:14:48.768808: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-12-22 09:14:48.769375: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: Quadro M1000M major: 5 minor: 0 memoryClockRate(GHz): 1.0715
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 2.08GiB
2017-12-22 09:14:48.769393: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Quadro M1000M, pci bus id: 0000:01:00.0, compute capability: 5.0)
/home/anton/usr-local/lib/python3.5/dist-packages/keras/models.py:252: UserWarning: No training configuration found in save file: the model was *not* compiled. Compile it manually.
  warnings.warn('No training configuration found in save file: '
/home/anton/CK_TOOLS/yad2k-1.0-linux-64/model_data/yolo.h5 model, anchors, and classes loaded.
2017-12-22 09:14:51.159534: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.73GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
Found 3 boxes for person.jpg
dog 0.79 (70, 258) (209, 356)
person 0.81 (190, 98) (271, 379)
horse 0.89 (399, 129) (605, 352)
Found 4 boxes for horses.jpg
horse 0.65 (0, 188) (169, 378)
horse 0.75 (253, 196) (435, 371)
horse 0.89 (435, 210) (603, 350)
horse 0.89 (7, 193) (305, 411)
Found 0 boxes for scream.jpg
Found 4 boxes for dog.jpg
motorbike 0.30 (60, 78) (113, 125)
dog 0.78 (137, 215) (323, 540)
truck 0.79 (462, 82) (694, 168)
bicycle 0.84 (81, 112) (554, 469)
Found 2 boxes for giraffe.jpg
zebra 0.83 (241, 208) (422, 442)
giraffe 0.89 (166, 0) (439, 448)
Found 1 boxes for eagle.jpg
bird 0.95 (128, 47) (643, 469)

Execution time: 5.627 sec.
```
