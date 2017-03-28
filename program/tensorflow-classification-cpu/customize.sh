#! /bin/bash

export CK_CUSTOM_LIBS="-lz"

export CK_LD_FLAGS_MISC="${CK_LD_FLAGS_MISC} -Wl,--allow-multiple-definition -Wl,--whole-archive"

# If not Android add pthread
if [ "${CK_ANDROID_ABI}" == "" ] ; then
  export CK_CUSTOM_LIBS="${CK_CUSTOM_LIBS} -lpthread"
fi

return 0
