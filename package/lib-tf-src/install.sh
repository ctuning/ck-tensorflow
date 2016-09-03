#! /bin/bash


# PACKAGE_DIR
# INSTALL_DIR
# TENSORFLOW_URL

export TENSORFLOW_SRC_DIR=${INSTALL_DIR}/src
export TENSORFLOW_OBJ_DIR=${INSTALL_DIR}/obj
export TENSORFLOW_INSTALL_DIR=${INSTALL_DIR}



echo ""
echo "Cloning TensorFlow from '${TENSORFLOW_URL}' ..."
rm -rf ${TENSORFLOW_SRC_DIR}
git clone ${TENSORFLOW_URL}  ${TENSORFLOW_SRC_DIR}
if [ "${?}" != "0" ] ; then
  echo "Error: Cloning TensorFlow from '${TENSORFLOW_URL}' failed!"
  exit 1
fi

cd $TENSORFLOW_SRC_DIR

$TENSORFLOW_SRC_DIR/configure

if [ "${?}" != "0" ] ; then
  echo "Error: TensorFlow installation configuration failed!"
  exit 1
fi

#Getting GPU_ENABLED answer
echo ""
DEFAULT="n"
read -e  -p "Have you chosen GPU enabled version? (y/[n]): " GPU_ENABLED
GPU_ENABLED="${GPU_ENABLED:-$DEFAULT}"
if [[ ($GPU_ENABLED != "n") && ($GPU_ENABLED != "y")]]
then
    echo "Error: GPU enabled answer is y or n."
    exit 1
fi



if [[ "$GPU_ENABLED" = "y" ]]
then
    sudo bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
else
    sudo bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
fi


sudo bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg


for pip_package in /tmp/tensorflow_pkg/*
do
    sudo pip install $pip_package
done

