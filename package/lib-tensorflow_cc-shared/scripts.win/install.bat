@echo off

rem
rem Installation script for CK packages.
rem
rem See CK LICENSE.txt for licensing details.
rem See CK Copyright.txt for copyright details.
rem
rem Developer(s): Grigori Fursin, 2016-2017
rem

if not "%CK_HAS_OPENMP%" == "0" (
  set CK_REF_LIBRARIES=%CK_LINKER_FLAG_OPENMP%
)

xcopy /E %ORIGINAL_PACKAGE_DIR%\*.patch %INSTALL_DIR%\obj

set JOB_COUNT=%CK_HOST_CPU_NUMBER_OF_PROCESSORS%

set CK_CMAKE_EXTRA=%CK_CMAKE_EXTRA% ^
 -DOPENCL_ROOT="%CK_ENV_LIB_OPENCL%" ^
 -DTUNERS=ON ^
 -DCLTUNE_ROOT:PATH="%CK_ENV_TOOL_CLTUNE%" ^
 -DCLIENTS=ON ^
 -DCBLAS_INCLUDE_DIRS:PATH="%CK_ENV_LIB_OPENBLAS_INCLUDE%" ^
 -DCBLAS_LIBRARIES:FILEPATH="%CK_ENV_LIB_OPENBLAS_LIB%\%CK_ENV_LIB_OPENBLAS_STATIC_NAME%" ^
 -DSAMPLES=ON ^
 -DCK_REF_LIBRARIES=%CK_REF_LIBRARIES%

exit /b 0
