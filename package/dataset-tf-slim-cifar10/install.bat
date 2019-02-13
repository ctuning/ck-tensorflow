@echo off

rem Installation script for the 2012 ImageNet Large Scale Visual Recognition
rem Challenge (ILSVRC'12) train dataset.
rem
rem See CK LICENSE.txt for licensing details.
rem See CK COPYRIGHT.txt for copyright details.
rem
rem Developer(s):
rem - Grigori Fursin, Grigori.Fursin@cTuning.org, 2018

rem PACKAGE_DIR
rem INSTALL_DIR

set IMAGENET_TRAIN_TAR=%INSTALL_DIR%\ILSVRC2012_img_train.tar

rem #####################################################################
echo.
echo Downloading archive ...

wget --no-check-certificate -c "%IMAGENET_TRAIN_URL%" -O "%IMAGENET_TRAIN_TAR%"

if %errorlevel% neq 0 (
 echo.
 echo Error: Failed downloading archive ...
 goto err
)

rem #####################################################################

echo.
echo Unpacking %DOWNLOAD_NAME% ...

cd /D %INSTALL_DIR%

tar xvf %IMAGENET_TRAIN_TAR%

rem if EXIST "%IMAGENET_TRAIN_TAR%" (
rem   del /Q /S %IMAGENET_TRAIN_TAR%
rem )

rem #####################################################################
echo.
echo Extracting individual classes ...

for %%f in ("n*.tar") do (
 mkdir %%~nf
 tar xvf %%f -C %%~nf
 del /Q %%~nf
)

rem #####################################################################
echo.
echo Successfully installed the ILSVRC'12 train dataset ...

exit /b 0

:err
exit /b 1
