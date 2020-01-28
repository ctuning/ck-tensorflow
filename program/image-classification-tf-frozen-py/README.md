# CPU execution

```bash
$ ck run program:image-classification-tf-frozen-py \
--dep_add_tags.weights=mobilenet,mlperf,v1-1.0-224,non-quantized \
--env.CUDA_VISIBLE_DEVICES=-1
```

# GPU execution with more available memory (33% by default)

```bash
$ ck run program:image-classification-tf-frozen-py \
--dep_add_tags.weights=mobilenet,mlperf,v1-1.0-224,non-quantized \
--env.CK_TF_GPU_MEMORY_PERCENT=80
```
