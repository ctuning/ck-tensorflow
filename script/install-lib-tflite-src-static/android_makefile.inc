# Set up Android building
ifeq ($(TARGET),ANDROID)
# Override NDK_ROOT on the command line with your own NDK location, e.g.
# make -f tensorflow/contrib/makefile/Makefile TARGET=ANDROID \
# NDK_ROOT=/path/to/your/ndk
# You need to have an Android version of the protobuf libraries compiled to link
# in. The compile_android_protobuf.sh script may help.

	ANDROID_HOST_OS_ARCH :=
	ifeq ($(HOST_OS),LINUX)
		ANDROID_HOST_OS_ARCH=linux
	endif
	ifeq ($(HOST_OS),OSX)
		ANDROID_HOST_OS_ARCH=darwin
	endif
	ifeq ($(HOST_OS),WINDOWS)
    $(error "windows is not supported.")
	endif

	ifeq ($(ARCH),x86_32)
		ANDROID_HOST_OS_ARCH := $(ANDROID_HOST_OS_ARCH)-x86
	else
		ANDROID_HOST_OS_ARCH := $(ANDROID_HOST_OS_ARCH)-$(ARCH)
	endif
    
	ifndef ANDROID_ARCH
		ANDROID_ARCH := armeabi-v7a
	endif

	ifeq ($(ANDROID_ARCH),arm64-v8a)
		TOOLCHAIN := aarch64-linux-android-4.9
		SYSROOT_ARCH := arm64
		BIN_PREFIX := aarch64-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),armeabi)
		TOOLCHAIN := arm-linux-androideabi-4.9
		SYSROOT_ARCH := arm
		BIN_PREFIX := arm-linux-androideabi
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),armeabi-v7a)
		TOOLCHAIN := arm-linux-androideabi-4.9
		SYSROOT_ARCH := arm
		BIN_PREFIX := arm-linux-androideabi
		MARCH_OPTION := -march=armv7-a -mfloat-abi=softfp -mfpu=neon
	endif
	ifeq ($(ANDROID_ARCH),mips)
		TOOLCHAIN := mipsel-linux-android-4.9
		SYSROOT_ARCH := mips
		BIN_PREFIX := mipsel-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),mips64)
		TOOLCHAIN := mips64el-linux-android-4.9
		SYSROOT_ARCH := mips64
		BIN_PREFIX := mips64el-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),x86)
		TOOLCHAIN := x86-4.9
		SYSROOT_ARCH := x86
		BIN_PREFIX := i686-linux-android
		MARCH_OPTION :=
	endif
	ifeq ($(ANDROID_ARCH),x86_64)
		TOOLCHAIN := x86_64-4.9
		SYSROOT_ARCH := x86_64
		BIN_PREFIX := x86_64-linux-android
		MARCH_OPTION :=
	endif
    
	ifndef NDK_ROOT
    $(error "NDK_ROOT is not defined.")
	endif
	CXX := $(CC_PREFIX) $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-g++
	CC := $(CC_PREFIX) $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-gcc
	CCFLAGS +=\
--sysroot $(NDK_ROOT)/platforms/android-$(ANDROID_API)/arch-$(SYSROOT_ARCH) \
-Wno-narrowing \
-fomit-frame-pointer \
$(MARCH_OPTION) \
-fPIE \
-fPIC
	CXXFLAGS +=\
--sysroot $(NDK_ROOT)/platforms/android-$(ANDROID_API)/arch-$(SYSROOT_ARCH) \
-Wno-narrowing \
-fomit-frame-pointer \
$(MARCH_OPTION) \
-fPIE \
-fPIC

	INCLUDES = \
-I$(NDK_ROOT) \
-I$(NDK_ROOT)/sysroot/usr/include \
-I$(NDK_ROOT)/sysroot/usr/include/$(BIN_PREFIX) \
-I$(NDK_ROOT)/sources/android/support/include \
-I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/include \
-I$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(ANDROID_ARCH)/include \
-I. \
-I$(MAKEFILE_DIR)/../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/farmhash/src \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(GENDIR)

	LIBS := \
-lstdc++ \
-lgnustl_static

	LD := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/$(BIN_PREFIX)/bin/ld

	LDFLAGS := \
$(MARCH_OPTION) \
-L$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(ANDROID_ARCH) \
-fPIE \
-pie \
-v

CCFLAGS+=-D__ANDROID_API__=$(ANDROID_API)
CXXFLAGS+=-D__ANDROID_API__=$(ANDROID_API)
LDFLAGS+=-D__ANDROID_API__=$(ANDROID_API)

	AR := $(NDK_ROOT)/toolchains/$(TOOLCHAIN)/prebuilt/$(ANDROID_HOST_OS_ARCH)/bin/$(BIN_PREFIX)-ar
	ARFLAGS := r
	LIBFLAGS += -Wl,--allow-multiple-definition -Wl,--whole-archive

endif  # ANDROID
