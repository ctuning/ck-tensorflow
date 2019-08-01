# Known issues
```
$ ck pull repo:ck-mlperf
$ ck install package:lib-tensorflow-1.14.0-src-static
$ ck compile program:image-classification-tf-cpp --dep_add_tags.library=v1.14
$ ck run program:image-classification-tf-cpp --dep_add_tags.weights=resnet
...
Batch 1 of 1
Batch loaded in 0.00210545 s
2019-08-01 11:12:56.505542: E tensorflow/core/common_runtime/executor.cc:641] Executor failed to create kernel. Not found: No registered '_FusedMatMul' OpKernel for CPU devices compatible with node {{node
 resnet_model/dense/BiasAdd}}
        .  Registered:  <no registered kernels>

         [[resnet_model/dense/BiasAdd]]
ERROR: Running model failed: Not found: No registered '_FusedMatMul' OpKernel for CPU devices compatible with node {{node resnet_model/dense/BiasAdd}}
        .  Registered:  <no registered kernels>

         [[resnet_model/dense/BiasAdd]]
```

See TensorFlow issues:
- https://github.com/tensorflow/tensorflow/issues/27671

No workaround is currently available.
