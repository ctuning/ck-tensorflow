# TBD: move model and dataset to CK!

mkdir -p data/
curl -L "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/multibox_detector/data/surfers.jpg" -o data/surfers.jpg
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip" -o mobile_multibox_v1a.zip
unzip mobile_multibox_v1a.zip -d data/
rm mobile_multibox_v1a.zip

ck compile program

ck run program
