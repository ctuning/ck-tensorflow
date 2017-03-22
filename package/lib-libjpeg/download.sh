#! /bin/bash

# CK download script for libjpeg package

# PACKAGE_DIR
# INSTALL_DIR
# LIBJPEG_URL

export LIBJPEG_INSTALL_DIR=${INSTALL_DIR}

######################################################################################
echo ""
echo "Cloning libjpeg from '${LIBJPEG_URL}' to '${LIBJPEG_INSTALL_DIR}' ..."
git clone ${LIBJPEG_URL}  ${LIBJPEG_INSTALL_DIR}
if [ "${?}" != "0" ] ; then
  echo "Error: Cloning libjpeg from '${LIBJPEG_URL}' failed!"
  exit 1
fi

exit 0
