#
# Installation script for CK packages.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developer(s): Grigori Fursin, 2017
#

# PACKAGE_DIR
# INSTALL_DIR

echo ""

SUDO="sudo "
if [[ ${CK_PYTHON_PIP_BIN_FULL} == /home/* ]] ; then
  SUDO=""
fi

read -r -p "Install OpenCV and other dependencies via sudo pip (Y/n)? " x

case "$x" in
  [nN][oO]|[nN])
    ;;
  *)
    ${SUDO} ${CK_ENV_COMPILER_PYTHON_FILE} -m pip install --upgrade opencv-python easydict image joblib
    ;;
esac
