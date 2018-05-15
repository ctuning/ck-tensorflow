@echo off

rem
rem Installation script for CK packages.
rem
rem See CK LICENSE.txt for licensing details.
rem See CK Copyright.txt for copyright details.
rem
rem Developer(s): Grigori Fursin, 2016-2017
rem

xcopy /E %ORIGINAL_PACKAGE_DIR%\*.patch %INSTALL_DIR%\obj

set JOB_COUNT=%CK_HOST_CPU_NUMBER_OF_PROCESSORS%

set CK_CMAKE_EXTRA=%CK_CMAKE_EXTRA% ^
  -DTENSORFLOW_SHARED=ON ^
  -DTENSORFLOW_STATIC=OFF ^
  -DCMAKE_CXX_COMPILER="%CK_CXX_PATH_FOR_CMAKE%" ^
  -DCMAKE_CXX_FLAGS="%CK_CXX_FLAGS_FOR_CMAKE%" ^
  -DCMAKE_CC_COMPILER="%CK_CC_PATH_FOR_CMAKE%" ^
  -DCMAKE_CC_FLAGS="%CK_CC_FLAGS_FOR_CMAKE%"

exit /b 0
