#! /bin/bash

#
# Installation script for Caffe.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developer(s):
# - Grigori Fursin, grigori.fursin@cTuning.org, 2016

# PACKAGE_DIR
# INSTALL_DIR

rm -f ${PACKAGE_NAME}*

################################################################################
echo ""
echo "Downloading package '${PACKAGE_CMD}' ..."

cd ${INSTALL_DIR}
bash -c "${PACKAGE_CMD}"
if [ "${?}" != "0" ] ; then
  echo "Error: error downloading package!"
  exit 1
fi

################################################################################
echo ""
echo "Unzipping package ${PACKAGE_NAME}.tar.gz ..."

gzip -d ${PACKAGE_NAME}.tar.gz
if [ "${?}" != "0" ] ; then
  echo "Error: Error unzipping '${PACKAGE_NAME}.tar.gz' failed!"
  exit 1
fi

################################################################################
echo ""
echo "Untarring package ${PACKAGE_NAME}.tar ..."

tar xvf ${PACKAGE_NAME}.tar
if [ "${?}" != "0" ] ; then
  echo "Error: Error untarring '${PACKAGE_NAME}.tar' failed!"
  exit 1
fi

rm -rf ${PACKAGE_NAME}.tar

