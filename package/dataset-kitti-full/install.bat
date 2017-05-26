@echo off

set KITTI_NAME=KITTI validation dataset
set KITTI_IMAGES_NAMES=%KITTI_NAME% images
set KITTI_LABELS_NAMES=%KITTI_NAME% labels

rem #####################################################################
echo.
echo "Downloading %KITTI_LABELS_NAMES% from '%KITTI_LABELS_URL%' ..."

wget --no-check-certificate -c "%KITTI_LABELS_URL%" -O "%KITTI_LABELS_ARCHIVE%"

echo.
echo "Unpacking '%KITTI_LABELS_ARCHIVE%' ..."
python -m zipfile -e "%KITTI_LABELS_ARCHIVE%" .
if EXIST "%KITTI_LABELS_ARCHIVE%" (
  del /Q /S "%KITTI_LABELS_ARCHIVE%"
)

rem #####################################################################
echo.
echo "Downloading %KITTI_IMAGES_NAMES% from '%KITTI_IMAGES_URL%' ..."

wget --no-check-certificate -c "%KITTI_IMAGES_URL%" -O "%KITTI_IMAGES_ARCHIVE%"

echo.
echo "Unpacking '%KITTI_IMAGES_ARCHIVE%' ..."
python -m zipfile -e "%KITTI_IMAGES_ARCHIVE%" .
if EXIST "%KITTI_IMAGES_ARCHIVE%" (
  del /Q /S "%KITTI_IMAGES_ARCHIVE%"
)

rem #####################################################################
echo.
echo "Successfully installed %KITTI_NAME% ..."
exit /b 0
