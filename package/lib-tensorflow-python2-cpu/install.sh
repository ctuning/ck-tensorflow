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

#Change configuration file and configure
cd $TENSORFLOW_SRC_DIR

cat $PACKAGE_DIR/export-variables |
while read line;
do
    sed -i "1 a $line" configure 
done 
 
sudo ./configure


if [ "${?}" != "0" ] ; then
  echo "Error: TensorFlow installation configuration failed!"
  exit 1
fi


#Create pip package and install 
sudo bazel build -c opt //tensorflow/tools/pip_package:build_pip_package

if [ "${?}" != "0" ] ; then
  echo "Error: Bazel building pip package failed"
  exit 1
fi

sudo bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

if [ "${?}" != "0" ] ; then
  echo "Error: Bazel building pip package failed"
  exit 1
fi

for pip_package in /tmp/tensorflow_pkg/*.whl
do
    sudo pip install $pip_package
done

