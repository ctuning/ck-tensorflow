#! /bin/bash

cd $INSTALL_DIR

rm -f ${MODEL_FILE}

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
echo "Unzipping file '${MODEL_FILE}' ..."
unzip ${MODEL_FILE}
if [ "${?}" != "0" ] ; then
  echo "Error: unzipping failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Removing file '${MODEL_FILE}' ..."
rm -f ${MODEL_FILE}

exit 0
