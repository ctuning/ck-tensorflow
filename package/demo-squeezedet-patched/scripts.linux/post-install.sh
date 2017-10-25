#
# Installation script for CK packages.
#
# See CK LICENSE.txt for licensing details.
# See CK Copyright.txt for copyright details.
#
# Developer(s): Grigori Fursin, 2017
#

# PACKAGE_DIR
# INSTALL_DIR

echo ""

read -r -p "Install OpenCV and other dependencies via sudo pip (Y/n)? " x

case "$x" in
  [nN][oO]|[nN])
    ;;
  *)
    sudo ${CK_PYTHON_PIP_BIN} install --upgrade opencv-python easydict image joblib
    ;;
esac
