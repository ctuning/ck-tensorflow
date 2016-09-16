#! /bin/bash


# PACKAGE_DIR
# INSTALL_DIR
# TENSORFLOW_URL

export TENSORFLOW_SRC_DIR=${CK_TOOLS}/tensorflow_src
export TENSORFLOW_LIB_DIR=${INSTALL_DIR}
export TENSORFLOW_INSTALL_DIR=${INSTALL_DIR}



echo ""
echo "Removing everything from '${TENSORFLOW_SRC_DIR}' and '${TENSORFLOW_LIB_DIR}'..."
sudo rm -rf ${TENSORFLOW_SRC_DIR}
sudo rm -rf ${TENSORFLOW_LIB_DIR}
echo ""
echo "Cloning TensorFlow from '${TENSORFLOW_URL}' to '${TENSORFLOW_SRC_DIR}' ..."
cd $CK_TOOLS
mkdir src
git clone ${TENSORFLOW_URL}  ${TENSORFLOW_SRC_DIR}
if [ "${?}" != "0" ] ; then
  echo "Error: Cloning TensorFlow from '${TENSORFLOW_URL}' failed!"
  exit 1
fi

#Change configuration file and configure
mkdir $TENSORFLOW_LIB_DIR
cd $TENSORFLOW_SRC_DIR

echo ""
echo "Editing configure file"
cat $PACKAGE_DIR/export-variables |
while read line;
do
    sed -i "1 a $line" configure 
done

sed -i "s#(./util/python/python_config.sh#echo $TENSORFLOW_LIB_DIR | (./util/python/python_config.sh#" configure

sudo -E ./configure

if [ "${?}" != "0" ] ; then
  echo "Error: TensorFlow installation configuration failed!"
  exit 1
fi



#Create pip package
if [ "$USE_CUDA" == 0 ]; then
    sudo bazel build -c opt //tensorflow/tools/pip_package:build_pip_package;
else 
    sudo bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package;
fi

if [ "${?}" != "0" ] ; then
  echo "Error: Bazel building pip package failed"
  exit 1
fi

sudo bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

if [ "${?}" != "0" ] ; then
  echo "Error: Bazel building pip package failed"
  exit 1
fi



#Install pip package
for pip_package in /tmp/tensorflow_pkg/*.whl
do
    if [ "$PYTHON3" == 0 ]; then
        pip install $pip_package -t $TENSORFLOW_LIB_DIR
    else
        pip3 install $pip_package -t $TENSORFLOW_LIB_DIR
    fi
done



sudo rm -rf $TENSORFLOW_SRC_DIR
