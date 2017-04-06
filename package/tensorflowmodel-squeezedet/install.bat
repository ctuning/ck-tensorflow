set SQDT_INSTALL=%INSTALL_DIR%\squeezeDet
set SQDT_ROOT=%SQDT_INSTALL%
set SQDT_DATA=%SQDT_ROOT%\data

rem ######################################################################################
echo.
echo "Removing everything from '%INSTALL_DIR%' ..."
rm -rf "%INSTALL_DIR%\*"
mkdir "%SQDT_INSTALL%"

rem ######################################################################################
echo. 
echo "Cloning SqueezeDet from '%SQUEEZEDET_URL%' to '%SQDT_INSTALL%' ..."
git clone "%SQUEEZEDET_URL%" "%SQDT_INSTALL%"
if %errorlevel% neq 0 (
  echo.
  echo "Error: Cloning SqueezeDet from '%SQUEEZEDET_URL%' failed!"
  goto err
)

rem ######################################################################################
echo.
echo "Downloading demo model parameters to '%SQDT_DATA%' ..."
wget --no-check-certificate -O "%SQDT_DATA%\model_checkpoints.tar.gz" "%DEMO_URL%"
if %errorlevel% neq 0 (
  echo.
  echo "Error: Downloading demo archive from '%DEMO_URL%' failed!"
  goto err
)

rem ######################################################################################
echo.
echo "Extracting demo archive '%SQDT_DATA%/model_checkpoints.tar.gz' ..."

cd "%SQDT_DATA%"
gzip -d "model_checkpoints.tar.gz" 
tar -xvf "model_checkpoints.tar"
if %errorlevel% neq 0 (
  echo.
  echo "Error: Extracting demo archive '%SQDT_DATA%\model_checkpoints.tar.gz' failed!"
  goto err
)
rem rm "%SQDT_DATA%\model_checkpoints.tgz"

rem ######################################################################################
echo.
echo "Downloading pretrained CNN models to '%SQDT_DATA%/' ..."

rem if [ "$SQUEEZENET_INCLUDED" != "0" ] ; then
rem   echo "Downloading pretrained SqueezeNet CNN archive from '${CNN_SQUEEZENET_URL}' ..."
rem   wget -P ${SQDT_DATA} ${CNN_SQUEEZENET_URL}
rem   if [ "${?}" != "0" ] ; then
rem     echo "Error: Downloading SqueezeNet CNN archive from '${CNN_SQUEEZENET_URL}' failed!"
rem     echo "To skip downloading this CNN archive, set the 'SQUEEZENET_INCLUDED' parameter in 'tensorflowmodel-squeezedet/.cm/meta.json' to 0."
rem     exit 1
rem   fi

rem   echo ""
rem   echo "Extracting '${SQDT_DATA}'/SqueezeNet.tgz ..."
rem   tar -xzvf ${SQDT_DATA}/SqueezeNet.tgz -C ${SQDT_DATA}/
rem   if [ "${?}" != "0" ] ; then
rem     echo "Error: Extracting CNN archive '${SQDT_DATA}'/SqueezeNet.tgz failed!"
rem     exit 1
rem   fi

rem   rm ${SQDT_DATA}/SqueezeNet.tgz
rem fi


rem if [ "$RESNET50_INCLUDED" != "0" ] ; then
rem   echo ""
rem   echo "Downloading pretrained ResNet50 CNN archive from '${CNN_RESNET50_URL}' ..."
rem   wget -P ${SQDT_DATA} ${CNN_RESNET50_URL}
rem   if [ "${?}" != "0" ] ; then
rem     echo "Error: Downloading ResNet50 CNN archive from '${CNN_RESNET_URL}' failed!"
rem     echo "To skip downloading this CNN archive, set the 'RESNET50_INCLUDED' parameter in 'tensorflowmodel-squeezedet/.cm/meta.json' to 0."
rem     exit 1
rem   fi

rem   echo ""
rem   echo "Extracting '${SQDT_DATA}'/ResNet.tgz ..."
rem   tar -xzvf ${SQDT_DATA}/ResNet.tgz -C ${SQDT_DATA}/
rem   if [ "${?}" != "0" ] ; then
rem     echo "Error: Extracting CNN archive '${SQDT_DATA}'/ResNet.tgz failed!"
rem     exit 1
rem   fi

rem   rm ${SQDT_DATA}/ResNet.tgz
rem fi

rem if [ "$VGG16_INCLUDED" != "0" ] ; then
rem   echo ""
rem   echo "Downloading pretrained VGG16 CNN archive from '${CNN_VGG16_URL}' ..."
rem   wget -P ${SQDT_DATA} ${CNN_VGG16_URL}
rem   if [ "${?}" != "0" ] ; then
rem     echo "Error: Downloading VGG16 CNN archive from '${CNN_VGG16_URL}' failed!"
rem     echo "To skip downloading this CNN archive, set the 'VGG16_INCLUDED' parameter in 'tensorflowmodel-squeezedet/.cm/meta.json' to 0."
rem     exit 1
rem   fi

rem   echo ""
rem   echo "Extracting '${SQDT_DATA}'/VGG16.tgz ..."
rem   tar -xzvf ${SQDT_DATA}/VGG16.tgz -C ${SQDT_DATA}/
rem   if [ "${?}" != "0" ] ; then
rem     echo "Error: Extracting CNN archive '${SQDT_DATA}'/VGG16.tgz failed!"
rem     exit 1
rem   fi

rem   rm ${SQDT_DATA}/VGG16.tgz
rem fi

rem ######################################################################################
rem # Some minor text editing to temporarily address point 1 in issue #10
rem # (https://github.com/ctuning/ck-tensorflow/issues/10).
rem echo.
rem echo "Making a minor source code change ..."
rem sed -i 's/reduction_indices/axis/g' ${SQDT_ROOT}/src/nn_skeleton.py
rem git -C ${SQDT_ROOT} diff

rem ######################################################################################
echo.
echo "Cleaning ..."
exit /b 0

:err
exit /b 1
