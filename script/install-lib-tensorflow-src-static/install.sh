#! /bin/bash

#
# Installation script for the TensorFlow library.
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
#
# This is the script for installing packages `lib-tensorflow-*-src-static`.
# It builds static TensorFlow library using provided scripts in
# `${CK-TOOLS}/lib-tensorflow-*-src-static/src/tensorflow/contrib/makefile/` directory.
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
  if [ -d ${ORIGINAL_PACKAGE_DIR}/patch.${CK_TARGET_OS_ID} ] ; then
    stage "Patching source directory ..."

    cd ${INSTALL_DIR}/${PACKAGE_SUB_DIR}

    for i in ${ORIGINAL_PACKAGE_DIR}/patch.${CK_TARGET_OS_ID}/*
    do
      echo "$i"
      patch -p1 < $i
      exit_if_error
    done
  fi
fi

echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "${CK_ANDROID_NDK_ROOT_DIR}"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
exit
if [[ "${CK_ANDROID_NDK_ROOT_DIR}" ]]; then

  # TODO: We have somehow to convert --target_os into an option supported by build_all_android.sh script:
  # arm64-v8a armeabi armeabi-v7a mips mips64 x86 x86_64 tegra
  TARGET_ARCH=arm64-v8a
  export NDK_ROOT=${CK_ANDROID_NDK_ROOT_DIR}
  ${INSTALL_DIR}/${PACKAGE_SUB_DIR}/tensorflow/contrib/makefile/build_all_android.sh -a ${TARGET_ARCH}
  
else

  ${INSTALL_DIR}/${PACKAGE_SUB_DIR}/tensorflow/contrib/makefile/build_all_linux.sh
  
fi

# TODO copy target files

return 0
