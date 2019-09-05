#! /bin/bash

########################################################################
echo
echo "Download weights from ${PACKAGE_URL} ..."
mkdir tmp
cd tmp
wget ${PACKAGE_URL}/${PACKAGE_NAME}

########################################################################
echo
echo "Unpack weights file ${PACKAGE_NAME} ..."
tar -zxvf ${PACKAGE_NAME}
mv ${PACKAGE_NAME1}/${FROZEN_GRAPH} ..
mv ${PACKAGE_NAME1}/${WEIGHTS_FILE}* ..
mv ${PACKAGE_NAME1}/${PIPELINE_CONFIG} ..

########################################################################
echo
echo "Remove temporary files ..."
cd ..
rm -rf tmp

#####################################################################
echo ""
#old version, based on the labelmaps from tf api folder
#echo "Copy label-map file ..."
#cp -f ${CK_ENV_TENSORFLOW_MODELS_OBJ_DET_DIR}/data/${LABELMAP_FILE} .

#new version, with labelmaps as packages
cp -f ${CK_ENV_LABELMAP_LABELMAP_FILE} .


#####################################################################
echo ""
echo "Copy custom model function hooks implementations ..."
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
