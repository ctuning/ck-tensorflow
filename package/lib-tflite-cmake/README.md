# TensorFlow Lite package

This package builds TensorFlow Lite using cmake from master (internally tagged `v2.3.99`).


## Installation

```
ck install package --tags=lib,tflite,v2.3.99
```

Optionally, to restrict the number of CPU cores used to build, append the following to the command (where 'X' is the number of processor cores).

```
-env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=X
```

