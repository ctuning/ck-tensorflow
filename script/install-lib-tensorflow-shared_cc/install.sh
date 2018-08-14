#! /bin/bash

#
# Installation script for the TensorFlow library.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#

function stage() {
  echo; echo "--------------------------------"; echo $1; echo
}

function exit_if_error() {
 if [ "${?}" != "0" ]; then exit 1; fi
}

function remove_dir_if_exists() {
  if [ -d $1 ]; then rm -rdf $1; fi
}

function patch_package() {
  if [ -d $1 ] ; then
    for i in $1/*.patch
    do
      echo "$i"
      patch -p1 < $i
      exit_if_error
    done
  fi
}


if [ "${PACKAGE_GIT}" == "YES" ] ; then
  stage "Cloning package ${PACKAGE_URL} ..."
  remove_dir_if_exists ${PACKAGE_SUB_DIR}
  git clone ${PACKAGE_GIT_CLONE_FLAGS} ${PACKAGE_URL} ${PACKAGE_SUB_DIR}
  exit_if_error

  stage "Checking out branch ${PACKAGE_GIT_CHECKOUT} ..."
  cd ${PACKAGE_SUB_DIR}
  git checkout ${PACKAGE_GIT_CHECKOUT}
  exit_if_error
fi


if [ "${PACKAGE_PATCH}" == "YES" ] ; then
  stage "Patching source directory ..."
  cd ${INSTALL_DIR}/${PACKAGE_SUB_DIR}
  patch_package ${ORIGINAL_PACKAGE_DIR}
  patch_package ${ORIGINAL_PACKAGE_DIR}/patch.${CK_TARGET_OS_ID}
fi  


stage "Clean existing installation ..."
remove_dir_if_exists ${INSTALL_DIR}/install
remove_dir_if_exists ${INSTALL_DIR}/obj


stage "Prepare custom build script ..."
ORIG_SCRIPT_DIR=${INSTALL_DIR}/${PACKAGE_SUB_DIR}/tensorflow_cc/cmake
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f ${ORIG_SCRIPT_DIR}/build_tensorflow.sh ]; then
  cp ${ORIG_SCRIPT_DIR}/build_tensorflow.sh ${ORIG_SCRIPT_DIR}/build_tensorflow.sh.orig
  cp ${THIS_SCRIPT_DIR}/build_tensorflow.sh ${ORIG_SCRIPT_DIR}
fi
if [ -f ${ORIG_SCRIPT_DIR}/build_tensorflow.sh.in ]; then
  cp ${ORIG_SCRIPT_DIR}/build_tensorflow.sh.in ${ORIG_SCRIPT_DIR}/build_tensorflow.sh.in.orig
  cp ${THIS_SCRIPT_DIR}/build_tensorflow.sh ${ORIG_SCRIPT_DIR}/build_tensorflow.sh.in
fi


stage "Run build script ..."
mkdir ${INSTALL_DIR}/obj
cd ${INSTALL_DIR}/obj
cp ${ORIGINAL_PACKAGE_DIR}/cmake-patches/*.patch .

if [ "${CK_AR_PATH_FOR_CMAKE}" != "" ] ; then
  XCMAKE_AR=" -DCMAKE_AR=${CK_AR_PATH_FOR_CMAKE} "
fi
if [ "${CK_RANLIB_PATH_FOR_CMAKE}" != "" ] ; then
  XCMAKE_RANLIB=" -DCMAKE_RANLIB=${CK_RANLIB_PATH_FOR_CMAKE} "
fi
if [ "${CK_LD_PATH_FOR_CMAKE}" != "" ] ; then
  XCMAKE_LD=" -DCMAKE_LINKER=${CK_LD_PATH_FOR_CMAKE} "
fi
  
cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/install" \
  -DCMAKE_C_COMPILER="${CK_CC_PATH_FOR_CMAKE}" \
  -DCMAKE_C_FLAGS="${CK_CC_FLAGS_FOR_CMAKE} ${CK_CC_FLAGS_ANDROID_TYPICAL} ${PACKAGE_FLAGS} ${CK_EXTRA_MISC_CC_FLAGS}" \
  -DCMAKE_CXX_COMPILER="${CK_CXX_PATH_FOR_CMAKE}" \
  -DCMAKE_CXX_FLAGS="${CK_CXX_FLAGS_FOR_CMAKE} ${CK_CXX_COMPILER_STDLIB} ${CK_CXX_FLAGS_ANDROID_TYPICAL} ${PACKAGE_FLAGS} ${CK_EXTRA_MISC_CXX_FLAGS}" \
  -DCMAKE_SHARED_LINKER_FLAGS="${CK_COMPILER_OWN_LIB_LOC}" \
  -DCMAKE_EXE_LINKER_FLAGS="${CK_COMPILER_OWN_LIB_LOC} ${CK_LINKER_FLAGS_ANDROID_TYPICAL} ${CK_EXTRA_MISC_LINKER_FLAGS}" \
  ${XCMAKE_AR} \
  ${XCMAKE_RANLIB} \
  ${XCMAKE_LD} \
  -DTENSORFLOW_SHARED=ON \
  -DTENSORFLOW_STATIC=OFF \
  ../${PACKAGE_SUB_DIR}/tensorflow_cc
exit_if_error

make
exit_if_error

echo "Install package ..."
make install
exit_if_error


return 0
