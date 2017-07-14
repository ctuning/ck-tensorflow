#! /bin/bash

# PACKAGE_DIR
# INSTALL_DIR
# PASCAL_ARCHIVE_NAME
# PASCAL_URL

#####################################################################
echo ""
echo "Downloading ${PASCAL_ARCHIVE_NAME} from '${PASCAL_URL}' ..."

wget ${PASCAL_URL} -P ${INSTALL_DIR}

if [ "${?}" != "0" ] ; then
  echo "Error: Downloading ${PASCAL_ARCHIVE_NAME} from '${PASCAL_URL}' failed!"
  exit 1
fi

######################################################################################
echo ""
echo "Extracting archive ..."
echo "Extracting ${PASCAL_ARCHIVE_NAME} ..."

tar -xf "${INSTALL_DIR}/${PASCAL_ARCHIVE_NAME}" -C ${INSTALL_DIR}

if [ "${?}" != "0" ] ; then
  echo "Error: Extracting ${PASCAL_ARCHIVE_NAME} failed!"
  exit 1
fi
rm "${INSTALL_DIR}/${PASCAL_ARCHIEVE_NAME}"

#####################################################################
echo ""
echo "Successfully installed Pascal VOC2012 dataset ..."
exit 0


exit 0

