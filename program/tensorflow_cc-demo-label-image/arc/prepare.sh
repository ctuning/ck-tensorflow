# TBD: move model and dataset to CK!

mkdir -p data/
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" | tar -C data/ -xz
curl -L "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/grace_hopper.jpg" -o data/grace_hopper.jpg

ck compile program

ck run program
