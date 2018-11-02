#! /bin/bash

# PACKAGE_DIR
# INSTALL_DIR

# KITTI_URL

KITTI_NAME="KITTI minimal dataset"
#####################################################################
echo ""
echo "Downloading ${KITTI_NAME} from '${KITTI_URL}' ..."

wget --no-check-certificate -c ${KITTI_URL} -O ${KITTI_ARCHIVE}

if [ "${?}" != "0" ] ; then
  echo "Error: Downloading ${KITTI_NAME} from '${KITTI_URL}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Calculating the MD5 hash of '${KITTI_ARCHIVE}' ..."
KITTI_MD5_CALC=($(md5sum ${KITTI_ARCHIVE}))
if [ "${?}" != "0" ] ; then
  echo "Error: Calculating the MD5 hash of '${KITTI_ARCHIVE}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Validating the MD5 hash of '${KITTI_ARCHIVE}' ..."
echo "Calculated MD5 hash: ${KITTI_MD5_CALC}"
echo "Reference MD5 hash: ${KITTI_MD5}"
if [ "${KITTI_MD5_CALC}" != "${KITTI_MD5}" ] ; then
  echo "Error: Validating the MD5 hash of '${KITTI_ARCHIVE}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Unpacking '${KITTI_ARCHIVE}' ..."

cd ${INSTALL_DIR}
tar xvf ${KITTI_ARCHIVE}
if [ "${?}" != "0" ] ; then
  echo "Error: Unpacking '${KITTI_ARCHIVE}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Deleting '${KITTI_ARCHIVE}' ..."

rm ${KITTI_ARCHIVE}
if [ "${?}" != "0" ] ; then
  echo "Error: Deleting '${KITTI_ARCHIVE}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Successfully installed ${KITTI_NAME} ..."
exit 0


exit 0
