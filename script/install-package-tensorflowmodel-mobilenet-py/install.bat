@echo off

rem
rem Copyright (c) 2018 cTuning foundation.
rem See CK COPYRIGHT.txt for copyright details.
rem
rem SPDX-License-Identifier: BSD-3-Clause.
rem See CK LICENSE.txt for licensing details.
rem
rem MobileNet for TensorFlow
rem Python model and weights install script
rem

set MULTIPLIER=%MODEL_MOBILENET_MULTIPLIER%
set RESOLUTION=%MODEL_MOBILENET_RESOLUTION%
set VERSION=%MODEL_MOBILENET_VERSION%

rem ########################################################################
echo.
echo Download weights from %PACKAGE_URL%/%PACKAGE_NAME% ...
wget %PACKAGE_URL%/%PACKAGE_NAME% --no-check-certificate

rem ########################################################################
echo.
echo Unpack weights file %PACKAGE_NAME% ...
gzip -d %PACKAGE_NAME%

set PACKAGE_NAME1=%PACKAGE_NAME:~0,-4%
echo Ungziped name: %PACKAGE_NAME1%.tar

tar xvf %PACKAGE_NAME1%.tar

rem ########################################################################
echo.
echo Remove temporary files ...
del /s %PACKAGE_NAME1%.tar

rem ########################################################################
echo.
echo Copy Python modules ...

set THIS_SCRIPT_DIR=%~dp0

echo Script directory: %THIS_SCRIPT_DIR%

xcopy /s /y %THIS_SCRIPT_DIR%mobilenet-model.py .

if "%VERSION%" == "1" (
 xcopy /s /y %THIS_SCRIPT_DIR%mobilenet_v1.py .
)

if "%VERSION%" == "2" (
  xcopy /s /y %THIS_SCRIPT_DIR%mobilenet_v2.py .
  xcopy /s /y %THIS_SCRIPT_DIR%mobilenet.py .
  xcopy /s /y %THIS_SCRIPT_DIR%conv_blocks.py .
)

exit /b 0
