# [TensorFlow Object Detection example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/multibox_detector) compiled with [TensorFlow_CC](https://github.com/FloopCZ/tensorflow_cc)

## Downloading the input data
```
$ cd `ck find program:tensorflow_cc-demo-multibox-detector`
$ mkdir -p data/
$ curl -L "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/multibox_detector/data/surfers.jpg" -o data/surfers.jpg
$ curl -L "https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip" -o mobile_multibox_v1a.zip
$ unzip mobile_multibox_v1a.zip -d data/
$ rm mobile_multibox_v1a.zip
```

## Compiling the example
```
$ ck compile program:tensorflow_cc-demo-multibox-detector
```

## Running the example
```
$ ck run program:tensorflow_cc-demo-multibox-detector
...
  (run ...)
executing code ...
2017-08-22 13:14:55.130004: I ../main.cc:287] Tensor<type: uint8 shape: [228,480,3] values: [[158 141 147]]...>
2017-08-22 13:14:55.130058: I ../main.cc:293] ===== Top 5 Detections ======
2017-08-22 13:14:55.130094: I ../main.cc:307] Detection 0: L:324.542 T:76.5764 R:373.26 B:214.957 (635) score: 0.267426
2017-08-22 13:14:55.130119: I ../main.cc:307] Detection 1: L:332.896 T:76.2751 R:372.116 B:204.614 (523) score: 0.245335
2017-08-22 13:14:55.130141: I ../main.cc:307] Detection 2: L:306.605 T:76.2228 R:371.356 B:217.32 (634) score: 0.21612
2017-08-22 13:14:55.130163: I ../main.cc:307] Detection 3: L:143.918 T:86.0909 R:187.333 B:195.885 (387) score: 0.171367
2017-08-22 13:14:55.130184: I ../main.cc:307] Detection 4: L:144.915 T:86.2675 R:185.243 B:165.246 (219) score: 0.169245
```
