#! /bin/bash

#
# CK installation script
#
# See CK LICENSE for licensing details.
# See CK COPYRIGHT for copyright details.
#

# PACKAGE_DIR
# INSTALL_DIR
# PYTHON_PACKAGE_NAME
# PIP_INSTALL_OPTIONS


    # This is where pip will install the modules.
    # It has its own funny structure we don't control :
    #
EXTRA_PYTHON_SITE=${INSTALL_DIR}/python_deps_site

SHORT_PYTHON_VERSION=`${CK_ENV_COMPILER_PYTHON_FILE} -c 'import sys;print(sys.version[:3])'`
export PACKAGE_LIB_DIR="${EXTRA_PYTHON_SITE}/lib/python${SHORT_PYTHON_VERSION}/site-packages"
export PYTHONPATH=$PACKAGE_LIB_DIR:$PYTHONPATH

echo "**************************************************************"
echo ""
echo "Cleanup: removing ${EXTRA_PYTHON_SITE}"
rm -rf "${EXTRA_PYTHON_SITE}"

######################################################################################
echo "Installing '${PYTHON_PACKAGE_NAME}' and its dependencies to '${PACKAGE_LIB_DIR}' ..."

${CK_ENV_COMPILER_PYTHON_FILE} -m pip install enum-compat ${PYTHON_PACKAGE_NAME}${PACKAGE_VERSION:+"==${PACKAGE_VERSION}"} --prefix=${EXTRA_PYTHON_SITE} ${PIP_INSTALL_OPTIONS}

if [ "${?}" != "0" ] ; then
  echo "Error: installation failed!"
  exit 1
fi

######################################################################################
MAJOR_PYTHON_VERSION=`${CK_ENV_COMPILER_PYTHON_FILE} -c 'import sys;print(sys.version[0])'`

if [ "$MAJOR_PYTHON_VERSION" == 3 ]; then
    echo ""
    echo "Conditionally cleaning up the 'enum34' package..."
    ${CK_ENV_COMPILER_PYTHON_FILE} -m pip uninstall enum34 ${SYS}
        #
        # enum34 may not have been installed, so we do not check the return code
        #
    echo ""
fi

    # Because we have to provide a fixed name via meta.json ,
    # and $PACKAGE_LIB_DIR depends on the Python version,
    # we solve it by creating a symbolic link with a fixed name.
    #
ln -s $PACKAGE_LIB_DIR ${INSTALL_DIR}/build
