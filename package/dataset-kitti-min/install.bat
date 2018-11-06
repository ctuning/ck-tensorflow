@echo off


set KITTI_NAME="KITTI minimal dataset"
rem #####################################################################
echo.
echo "Downloading %KITTI_NAME% from '%KITTI_URL%' ..."

wget --no-check-certificate -c "%KITTI_URL%" -O "%KITTI_ARCHIVE%"

if %errorlevel% neq 0 (
  echo.
  echo Error: Failed downloading archive ...
  goto err
)

rem #####################################################################
echo.
echo "Unpacking '%KITTI_ARCHIVE%' ..."

cd /D "%INSTALL_DIR%"
tar xvf "%KITTI_ARCHIVE%"

if EXIST "%KITTI_ARCHIVE%" (
  del /Q /S "%KITTI_ARCHIVE%"
)

rem #####################################################################
echo.
echo "Successfully installed %KITTI_NAME% ..."
exit /b 0

:err
exit /b 1
