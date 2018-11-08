@echo off

rem
rem Installation script for CK packages.
rem
rem See CK LICENSE.txt for licensing details.
rem See CK COPYRIGHT.txt for copyright details.
rem
rem Developer(s): Grigori Fursin, 2017
rem

rem PACKAGE_DIR
rem INSTALL_DIR

%CK_PYTHON_PIP_BIN% install --upgrade opencv-python easydict image joblib

rem Sometimes issues with OpenCV on Anaconda
if NOT "%CK_CONDA_BIN_FULL%" == "" (
 %CK_PYTHON_PIP_BIN% uninstall opencv-python
 %CK_CONDA_BIN_FULL% install -c conda-forge opencv
)
 
