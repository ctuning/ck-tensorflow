#! /bin/bash


#Getting python version from user
DEFAULT="2"
read -e  -p "Choose python version (3/[2]): " PYTHON
PYTHON="${PYTHON:-$DEFAULT}"
if [[ ($PYTHON != "2") && ($PYTHON != "3")]]
then
    echo "Error: version ${PYTHON} doesn't exist. Choose from [2/3]"
    exit 1
fi



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



echo "" 
echo  "TensorFlow installation started"



if [[ "$PYTHON" = "2" ]]
then 
    if [[ "$GPU_ENABLED" = "y" ]]
    then
         export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
    else
          export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
    fi

sudo pip install --upgrade $TF_BINARY_URL

elif [[ "$PYTHON" -eq "3" ]]
then
    if [[ "$GPU_ENABLED" = "y" ]]
    then
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
    else
        export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
    fi

sudo pip3 install --upgrade $TF_BINARY_URL

fi

