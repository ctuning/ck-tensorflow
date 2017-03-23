#! /bin/bash

#
# Installation script.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developer(s):
# - Grigori Fursin, grigori@dividiti.com, 2016
#

# PACKAGE_DIR
# INSTALL_DIR

################################################################################
echo ""
echo "Cloning from '${PACKAGE_URL}' ..."

cd ${INSTALL_DIR}

UNI_PACKAGE_INSTALL=${PACKAGE_INSTALL}

if [[ "$OSTYPE" == "darwin"* ]]; then
  UNI_PACKAGE_INSTALL=${PACKAGE_INSTALL_OSX}
fi

rm -rf ${UNI_PACKAGE_INSTALL}
wget ${PACKAGE_URL}/${UNI_PACKAGE_INSTALL}

if [ "${?}" != "0" ] ; then
  echo "Error: downloading failed!"
  exit 1
fi

chmod 755 ${UNI_PACKAGE_INSTALL}

./${UNI_PACKAGE_INSTALL} --prefix=${INSTALL_DIR} --bazelrc=${INSTALL_DIR}
if [ "${?}" != "0" ] ; then
  echo "Error: executing installer failed!"
  exit 1
fi

exit 0
