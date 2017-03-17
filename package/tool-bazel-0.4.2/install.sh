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

if [[ "$OSTYPE" == "darwin"* ]]; then
  rm -rf ${PACKAGE_INSTALL_OSX}
  wget ${PACKAGE_URL}/${PACKAGE_INSTALL_OSX}
else
  rm -rf ${PACKAGE_INSTALL}
  wget ${PACKAGE_URL}/${PACKAGE_INSTALL}
fi

if [ "${?}" != "0" ] ; then
  echo "Error: downloading failed!"
  exit 1
fi

chmod 755 ${PACKAGE_INSTALL}

./${PACKAGE_INSTALL} --prefix=${INSTALL_DIR} --bazelrc=${INSTALL_DIR}
if [ "${?}" != "0" ] ; then
  echo "Error: executing installer failed!"
  exit 1
fi

exit 0
