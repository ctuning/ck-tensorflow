@echo off

rem  CK installation script for TensorFlow package
rem
rem Developer(s):
rem  * Grigori Fursin, dividiti/cTuning foundation
rem

set TENSORFLOW_LIB_DIR=%INSTALL_DIR%\lib

rem ######################################################################################
echo.
echo Downloading and installing misc deps ...
echo.

rem %CK_PYTHON_PIP_BIN% install --upgrade pip
rem %CK_PYTHON_PIP_BIN% install easydict joblib image numpy scipy enum-compat
rem %CK_PYTHON_PIP_BIN% uninstall enum34

%CK_ENV_COMPILER_PYTHON_FILE% -m pip install --ignore-installed protobuf easydict joblib image numpy scipy enum-compat -t %TENSORFLOW_LIB_DIR%
if %errorlevel% neq 0 (
 echo.
 echo Error: Failed installing misc deps ...
 exit /b 1
)

echo.
echo Downloading and installing TensorFlow prebuilt binaries (%TF_PYTHON_URL%) ...
echo.

%CK_ENV_COMPILER_PYTHON_FILE% -m pip install --ignore-installed %TF_PYTHON_URL% -t %TENSORFLOW_LIB_DIR% --trusted-host storage.googleapis.com
if %errorlevel% neq 0 (
 echo.
 echo Error: Failed installing TensorFlow prebuilt binaries
 exit /b 1
)

cd /D %INSTALL_DIR%\lib
rmdir /Q /S enum

if %errorlevel% neq 0 (
 echo.
 echo Error: Failed instaling TF ...
 exit /b 1
)

exit /b 0
