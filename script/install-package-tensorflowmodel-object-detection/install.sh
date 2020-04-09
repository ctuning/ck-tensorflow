#! /bin/bash

########################################################################
echo
echo "Download weights from ${PACKAGE_URL} ..."
mkdir tmp
cd tmp
wget ${PACKAGE_URL}/${PACKAGE_NAME}

########################################################################
if [ ${PACKAGE_UNZIP} == "YES" ] && [ ${PACKAGE_UNTAR} == "YES" ]; then
echo
echo "Unpack weights file ${PACKAGE_NAME} ..."
tar -zxvf ${PACKAGE_NAME}
mv ${PACKAGE_NAME1}/${FROZEN_GRAPH} ..
mv ${PACKAGE_NAME1}/${WEIGHTS_FILE}* ..
mv ${PACKAGE_NAME1}/${PIPELINE_CONFIG} ..
fi

#####################################################################
if [ -f ${FROZEN_GRAPH} ] && [ ${FROZEN_GRAPH} != "graph.pb" ]; then
  mv ${FROZEN_GRAPH} ..
  cd ..
  rm -f graph.pb
  ln -s ${FROZEN_GRAPH} graph.pb
  cd tmp/
fi

########################################################################
echo
echo "Remove temporary files ..."
cd ..
rm -rf tmp

#####################################################################
echo ""
echo "Copy label-map file from '${CK_ENV_LABELMAP_FILE}' ..."
cp -f ${CK_ENV_LABELMAP_FILE} .

#####################################################################
if [ ! -f ${PIPELINE_CONFIG} ];  then
  echo ""
  echo "Copy '${PIPELINE_CONFIG}' from '${ORIGINAL_PACKAGE_DIR}' ..."
  cp -f ${ORIGINAL_PACKAGE_DIR}/${PIPELINE_CONFIG} .
fi

#####################################################################
echo ""
echo "Copy custom model hooks (if any) ..."
echo "${ORIGINAL_PACKAGE_DIR}"
echo "to"
echo "$PWD"

FILE=${ORIGINAL_PACKAGE_DIR}/custom_hooks.py
if [ -f "$FILE" ]; then
    cp -f "$FILE" .
fi

FILE=${ORIGINAL_PACKAGE_DIR}/custom_tensorRT.py
if [ -f "$FILE" ]; then
    cp -f "$FILE" .
fi

#cp -f ${ORIGINAL_PACKAGE_DIR}/custom_hooks.py .
#cp -f ${ORIGINAL_PACKAGE_DIR}/custom_tensorRT.py .

#####################################################################
echo ""
echo "Successfully installed '${MODEL_NAME}' TensorFlow model ..."
exit 0
