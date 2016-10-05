#! /bin/bash



sudo rm-rf $INSTALL_DIR
mkdir $INSTALL_DIR



echo "" 
echo  "TensorFlow installation started"


if [[ $PYTHON3 == 0 ]]
then
    if [[ $GPU_ENABLED == 1 ]]
    then
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl 
    else
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl 
    fi
    sudo pip install --upgrade $TF_BINARY_URL -t $INSTALL_DIR
else
    if [[ $GPU_ENABLED == 1 ]]
    then
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
    else
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
    fi
    sudo pip3 install --upgrade $TF_BINARY_URL -t $INSTALL_DIR
fi




