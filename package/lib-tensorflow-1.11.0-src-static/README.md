# Known issues

```
$ ck install package:lib-tensorflow-1.11.0-src-static
$ ck compile program:image-classification-tf-cpp
...
g++-7 -c    -I../ -DCK_HOST_OS_NAME2_LINUX=1 -DCK_HOST_OS_NAME_LINUX=1 -DCK_TARGET_OS_NAME2_LINUX=1 -DCK_TARGET_OS_NAME_LINUX=1 -std=c++11 -I/home/anton/CK_TOOLS/lib-rtl-xopenme-0.3-gcc-8.2.0-linux-64/include -I/home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src -I/home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/contrib/makefile/downloads/protobuf/src -I/home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/contrib/makefile/downloads -I/home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/contrib/makefile/downloads/eigen -I/home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/contrib/makefile/gen/proto -I/home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/contrib/makefile/downloads/nsync/public  ../classification.cpp  -o classification.o
In file included from /home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/core/platform/tensor_coding.h:22:0,
                 from /home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/core/framework/resource_handle.h:19,
                 from /home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/core/framework/allocator.h:24,
                 from /home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/core/framework/tensor.h:20,
                 from /home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/core/public/session.h:24,
                 from ../classification.cpp:12:
/home/anton/CK_TOOLS/lib-tensorflow-src-static-1.11.0-gcc-7.3.0-linux-64/src/tensorflow/core/lib/core/stringpiece.h:34:10: fatal error: absl/strings/string_view.h: No such file or directory
 #include "absl/strings/string_view.h"
          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
***************************************************************************************
Compilation time: 0.186 sec.; Object size: 0; Total binary size: 0; MD5:
Warning: compilation failed!
```

See TensorFlow issues:
- https://github.com/tensorflow/tensorflow/issues/22240
- https://github.com/tensorflow/tensorflow/issues/22320

No workaround is currently available.
