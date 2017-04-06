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

if %SQUEEZENET_INCLUDED% neq 0 (
  echo "Downloading pretrained SqueezeNet CNN archive from '%CNN_SQUEEZENET_URL%' ..."
  wget --no-check-certificate -O "SqueezeNet.tar.gz" "%CNN_SQUEEZENET_URL%"
  gzip -d "SqueezeNet.tar.gz" 
  tar -xvf "SqueezeNet.tar"
)

if %RESNET50_INCLUDED% neq 0 (
  echo "Downloading pretrained ResNet50 CNN archive from '%CNN_RESNET50_URL%' ..."
  wget --no-check-certificate -O "ResNet.tar.gz" "%CNN_RESNET50_URL%"
  gzip -d "ResNet.tar.gz" 
  tar -xvf "ResNet.tar"
)

if %VGG16_INCLUDED% neq 0 (
  echo "Downloading pretrained VGG16 CNN archive from '%CNN_VGG16_URL%' ..."
  wget --no-check-certificate -O "VGG16.tar.gz" "%CNN_VGG16_URL%"
  gzip -d "VGG16.tar.gz" 
  tar -xvf "VGG16.tar"
)

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
