#! /bin/bash

#
# Download script for TensorFlow model weights.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developer(s):
# - Anton Lokhmotov, anton@dividiti.com, 2016
# - Grigori Fursin, grigori@dividiti.com, 2016

# ORIGINAL_PACKAGE_DIR (path to original package even if scripts are used from some other package or script)
# PACKAGE_DIR (path where scripts are reused)
# INSTALL_DIR

#####################################################################
cd $INSTALL_DIR

rm -f ${MODEL_FILE1}
rm -f ${MODEL_FILE2}

#####################################################################
echo ""
echo "Downloading the weights from '${MODEL_URL}' ..."
wget -c ${MODEL_URL}
if [ "${?}" != "0" ] ; then
  echo "Error: Downloading the weights from '${MODEL_URL}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Ungzipping file '${MODEL_FILE1}' ..."
gzip -d ${MODEL_FILE1}
if [ "${?}" != "0" ] ; then
  echo "Error: ungzipping failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Untarring file '${MODEL_FILE2}' ..."
tar xvf ${MODEL_FILE2}
if [ "${?}" != "0" ] ; then
  echo "Error: ungzipping failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Removing file '${MODEL_FILE2}' ..."
rm -f ${MODEL_FILE2}

exit 0
