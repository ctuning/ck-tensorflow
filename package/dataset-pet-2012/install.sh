#! /bin/bash

# PACKAGE_DIR
# INSTALL_DIR

# PET_URL
# PET_IMAGES_URL
# PET_LABELS_URL
# PET_IMAGES_ARCHIVE
# PET_LABELS_ARCHIVE

PET_NAME="PET OXFORD-IIIT 2012 dataset"

cd ${INSTALL_DIR}

#####################################################################
echo ""
echo "Downloading ${PET_NAME} images from '${PET_IMAGES_URL}' ..."

wget ${PET_IMAGES_URL} 

if [ "${?}" != "0" ] ; then
  echo "Error: Downloading ${PET_NAME} images from '${PET_IMAGES_URL}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Downloading ${PET_NAME} labels from '${PET_LABELS_URL}' ..."

wget ${PET_LABELS_URL} 

if [ "${?}" != "0" ] ; then
  echo "Error: Downloading ${PET_NAME} labels from '${PET_LABELS_URL}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Unpacking '${PET_IMAGES_ARCHIVE}' ..."

tar -xvf ${PET_IMAGES_ARCHIVE}
if [ "${?}" != "0" ] ; then
  echo "Error: Unpacking '${PET_IMAGES_ARCHIVE}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Unpacking '${PET_LABELS_ARCHIVE}' ..."

tar -xvf ${PET_LABELS_ARCHIVE}
if [ "${?}" != "0" ] ; then
  echo "Error: Unpacking '${PET_LABELS_ARCHIVE}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Deleting '${PET_IMAGES_ARCHIVE}' ..."

rm ${PET_IMAGES_ARCHIVE}
if [ "${?}" != "0" ] ; then
  echo "Error: Deleting '${PET_IMAGES}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Deleting '${PET_LABELS_ARCHIVE}' ..."

rm ${PET_LABELS_ARCHIVE}
if [ "${?}" != "0" ] ; then
  echo "Error: Deleting '${PET_IMAGES}' failed!"
  exit 1
fi

#####################################################################
echo ""
echo "Successfully installed '${PET_NAME}'"
exit 0

