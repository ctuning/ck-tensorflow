#! /bin/bash



sudo rm-rf $INSTALL_DIR
mkdir $INSTALL_DIR

#Getting GPU_ENABLED answer
echo ""
DEFAULT="y"
read -e  -p "GPU enabled? (n/[y]): " GPU_ENABLED
GPU_ENABLED="${GPU_ENABLED:-$DEFAULT}"
if [[ ($GPU_ENABLED != "n") && ($GPU_ENABLED != "y")]]
then
    echo "Error: GPU enabled answer is y or n."
    exit 1
fi



if [[ "$GPU_ENABLED" = "y" ]]
    then
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
    else
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
fi



sudo pip3 install --upgrade $TF_BINARY_URL -t $INSTALL_DIR

