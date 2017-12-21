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


# Run YAD2K demo

```
$ ck run program:yad2k-demo --cmd_key=test
```
