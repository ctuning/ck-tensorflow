# [TensorFlow Image Recognition example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image) compiled with [TensorFlow_CC](https://github.com/FloopCZ/tensorflow_cc)

## Compiling the example
```
$ ck compile program:tensorflow_cc-demo-label-image
```

## Running the example
```
$ ck run program:tensorflow_cc-demo-label-image
...
  (run ...)
executing code ...
2017-08-22 12:54:04.335843: I ../main.cc:213] military uniform (653): 0.834306
2017-08-22 12:54:04.335893: I ../main.cc:213] mortarboard (668): 0.0218693
2017-08-22 12:54:04.335900: I ../main.cc:213] academic gown (401): 0.010358
2017-08-22 12:54:04.335904: I ../main.cc:213] pickelhaube (716): 0.00800817
2017-08-22 12:54:04.335908: I ../main.cc:213] bulletproof vest (466): 0.00535091
```

## Not needed anymore - CK will handle that: downloading the input data
```
$ cd `ck find program:tensorflow_cc-demo-label-image`
$ mkdir -p data/
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -C data/ -xz
$ curl -L "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/grace_hopper.jpg" -o data/grace_hopper.jpg
```
