# src/Makefile - Standalone build for RawHash2
#
# Defaults: POD5=enabled, HDF5=disabled, SLOW5=disabled
#
# Options:
#   DEBUG=1     Debug build (-O2 -fsanitize=address -g)
#   PROFILE=1   Profiling build (-g -fno-omit-frame-pointer -DPROFILERH=1)
#   NOPOD5=1    Disable POD5 support
#   NOHDF5=0    Enable HDF5/FAST5 support (disabled by default)
#   NOSLOW5=0   Enable SLOW5/BLOW5 support (disabled by default)
#   asan=1      Enable AddressSanitizer
#   tsan=1      Enable ThreadSanitizer

# =====================================================================
# Default feature toggles (match CMake defaults)
# POD5 is ON by default (set NOPOD5=1 to disable)
# HDF5 is OFF by default (set NOHDF5= or NOHDF5=0 to enable)
# SLOW5 is OFF by default (set NOSLOW5= or NOSLOW5=0 to enable)
# =====================================================================
NOHDF5  ?= 1
NOSLOW5 ?= 1

# =====================================================================
# System detection
# =====================================================================
SYSTEM_NAME      ?= $(shell uname -s)
SYSTEM_PROCESSOR ?= $(shell uname -m)

# =====================================================================
# Directories
# =====================================================================
SRCDIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
WORKDIR := $(abspath $(SRCDIR)/../extern)

# =====================================================================
# Compiler selection
# When POD5 or HDF5 is enabled, ALL .c files must be compiled as C++:
#   - rsig.c contains C++ code (std::string, new, etc.) in #ifndef NHDF5RH
#   - POD5 static libs (libarrow.a) require C++ linker
# When neither is enabled, pure C compilation suffices.
# =====================================================================
NEED_CXX := 0
ifneq ($(NOPOD5),1)
    NEED_CXX := 1
endif
ifneq ($(NOHDF5),1)
    NEED_CXX := 1
endif

ifeq ($(NEED_CXX),1)
    COMPILER    = $(CXX)
    LANG_STD    = -std=c++11
    LANG_MODE   = -x c++
    COMPAT_WARN =
else
    COMPILER    = $(CC)
    LANG_STD    = -std=c99
    LANG_MODE   =
    COMPAT_WARN = -Wc++-compat
endif

# =====================================================================
# Base compiler flags
# =====================================================================
BASE_DEFS = -DHAVE_KALLOC

ifdef DEBUG
    OPT_FLAGS = -O2 -fsanitize=address -g
else
    OPT_FLAGS = -O3
endif

ifdef PROFILE
    OPT_FLAGS += -g -fno-omit-frame-pointer
    BASE_DEFS += -DPROFILERH=1
endif

# =====================================================================
# Architecture-specific flags
# =====================================================================
ARCH_FLAGS =
ARCH_DEFS  =

# --- x86 / x86_64 ---
ifneq ($(filter x86_64 i686 i386,$(SYSTEM_PROCESSOR)),)
    ARCH_FLAGS += -march=native
    # Detect immintrin.h availability
    HAVE_IMMINTRIN := $(shell echo '\#include <immintrin.h>' | \
        $(if $(filter 1,$(NEED_CXX)),$(CXX),$(CC)) -x c -fsyntax-only - 2>/dev/null && echo 1 || echo 0)
    ifeq ($(HAVE_IMMINTRIN),1)
        ARCH_DEFS += -DHAVE_IMMINTRIN_H
    endif
endif

# --- AArch64 (arm64) ---
ifneq ($(filter aarch64 arm64 ARM64,$(SYSTEM_PROCESSOR)),)
    ARCH_FLAGS += -fsigned-char
    ARCH_DEFS  += -D_FILE_OFFSET_BITS=64
    # -march=native may not work on all Apple Silicon toolchains
    MARCH_OK := $(shell $(if $(filter 1,$(NEED_CXX)),$(CXX),$(CC)) -march=native -x c -c /dev/null -o /dev/null 2>/dev/null && echo 1 || echo 0)
    ifeq ($(MARCH_OK),1)
        ARCH_FLAGS += -march=native
    endif
endif

# --- ARM 32-bit ---
ifneq ($(filter arm armv7l armv6l,$(SYSTEM_PROCESSOR)),)
    ARCH_FLAGS += -fsigned-char -mfpu=neon
    ARCH_DEFS  += -D_FILE_OFFSET_BITS=64
    MARCH_OK := $(shell $(CC) -march=native -x c -c /dev/null -o /dev/null 2>/dev/null && echo 1 || echo 0)
    ifeq ($(MARCH_OK),1)
        ARCH_FLAGS += -march=native
    endif
endif

# =====================================================================
# Sanitizer flags (standalone, outside DEBUG)
# =====================================================================
ifneq ($(asan),)
    OPT_FLAGS     += -fsanitize=address
    EXTRA_LDFLAGS += -fsanitize=address
endif

ifneq ($(tsan),)
    OPT_FLAGS     += -fsanitize=thread
    EXTRA_LDFLAGS += -fsanitize=thread
endif

# =====================================================================
# Combined compiler flags
# =====================================================================
ALL_FLAGS = $(LANG_STD) -Wall $(COMPAT_WARN) $(OPT_FLAGS) $(ARCH_FLAGS) \
            $(BASE_DEFS) $(ARCH_DEFS) -pthread

# =====================================================================
# POD5 configuration (v0.3.36)
# =====================================================================
POD5_VERSION = 0.3.36
POD5_REPO    = https://github.com/nanoporetech/pod5-file-format

ifneq ($(NOPOD5),1)
    # Platform-specific URL and library directory
    ifeq ($(SYSTEM_NAME),Linux)
        ifneq ($(filter aarch64 arm64 ARM64,$(SYSTEM_PROCESSOR)),)
            POD5_ARCHIVE = lib_pod5-$(POD5_VERSION)-linux-arm64.tar.gz
        else
            POD5_ARCHIVE = lib_pod5-$(POD5_VERSION)-linux-x64.tar.gz
        endif
        POD5_LIB_SUBDIR      = lib64
        POD5_EXTRA_LIBS_FILE = libjemalloc_pic.a
    else ifeq ($(SYSTEM_NAME),Darwin)
        ifneq ($(filter arm64 ARM64 aarch64,$(SYSTEM_PROCESSOR)),)
            POD5_ARCHIVE = lib_pod5-$(POD5_VERSION)-osx-14.0-arm64.tar.gz
        else
            $(error POD5 $(POD5_VERSION) has no pre-built binaries for macOS x86_64. Use NOPOD5=1.)
        endif
        POD5_LIB_SUBDIR      = lib
        POD5_EXTRA_LIBS_FILE =
    endif

    POD5_URL = $(POD5_REPO)/releases/download/$(POD5_VERSION)/$(POD5_ARCHIVE)
    POD5_DIR = $(WORKDIR)/pod5-$(POD5_VERSION)-$(SYSTEM_NAME)

    # Allow user to override include/lib paths (skip download if set)
    ifdef POD5_INCLUDE_DIR
        POD5_DOWNLOAD = 0
    endif
    POD5_INCLUDE_DIR ?= $(POD5_DIR)/include
    POD5_LIB_DIR     ?= $(POD5_DIR)/$(POD5_LIB_SUBDIR)

    POD5_LIBRARIES = $(POD5_LIB_DIR)/libpod5_format.a \
                     $(POD5_LIB_DIR)/libarrow.a
    ifneq ($(POD5_EXTRA_LIBS_FILE),)
        POD5_LIBRARIES += $(POD5_LIB_DIR)/$(POD5_EXTRA_LIBS_FILE)
    endif

    INCLUDES += -I$(POD5_INCLUDE_DIR)
    LIBS     += $(POD5_LIBRARIES)
else
    ALL_FLAGS += -DNPOD5RH=1
endif

# =====================================================================
# HDF5 configuration
# =====================================================================
HDF5_DIR       ?= $(WORKDIR)/hdf5
HDF5_BUILD_DIR ?= $(HDF5_DIR)/build

ifdef HDF5_INCLUDE_DIR
    HDF5_COMPILE = 0
endif
HDF5_INCLUDE_DIR ?= $(HDF5_BUILD_DIR)/include
HDF5_LIB_DIR     ?= $(HDF5_BUILD_DIR)/lib
HDF5_LIB         ?= hdf5

ifneq ($(NOHDF5),1)
    INCLUDES += -I$(HDF5_INCLUDE_DIR)
    LIBS     += $(HDF5_LIB_DIR)/lib$(HDF5_LIB).a
else
    ALL_FLAGS += -DNHDF5RH=1
endif

# =====================================================================
# SLOW5 configuration
# =====================================================================
SLOW5_DIR ?= $(WORKDIR)/slow5lib

ifdef SLOW5_INCLUDE_DIR
    SLOW5_COMPILE = 0
endif
SLOW5_INCLUDE_DIR ?= $(SLOW5_DIR)/include
SLOW5_LIB_DIR     ?= $(SLOW5_DIR)/lib

ifneq ($(NOSLOW5),1)
    INCLUDES += -I$(SLOW5_INCLUDE_DIR)
    LIBS     += $(SLOW5_LIB_DIR)/libslow5.a
else
    ALL_FLAGS += -DNSLOW5RH=1
endif

# gRPC is not supported in standalone builds (requires CMake).
ALL_FLAGS += -DNGRPCRH=1

# =====================================================================
# zstd configuration (needed by both POD5 and SLOW5)
# =====================================================================
ZSTD_STATIC =
ifneq ($(NOPOD5),1)
    ZSTD_STATIC = $(WORKDIR)/zstd/lib/libzstd.a
endif
ifneq ($(NOSLOW5),1)
    ZSTD_STATIC = $(WORKDIR)/zstd/lib/libzstd.a
endif
LIBS += $(ZSTD_STATIC)

# =====================================================================
# System libraries (must come after static libs for correct link order)
# =====================================================================
LIBS += -lm -lz $(EXTRA_LDFLAGS)
ifeq ($(SYSTEM_NAME),Linux)
    LIBS += -ldl
endif

# =====================================================================
# C++ runtime libraries (must come LAST for correct link ordering)
# POD5 v0.3.36 static archives (libarrow.a, libpod5_format.a) reference
# C++ stdlib symbols that must be resolved after those archives.
# =====================================================================
ifeq ($(NEED_CXX),1)
    ifeq ($(SYSTEM_NAME),Linux)
        # POD5 v0.3.36 uses std::filesystem; -lstdc++fs provides these
        # symbols on GCC < 9 and is a harmless stub on GCC 9+.
        LIBS += -lstdc++fs -lstdc++
    endif
endif

# =====================================================================
# Source files and objects
# =====================================================================
SRCS = kthread.c kalloc.c bseq.c roptions.c sequence_until.c rutils.c \
       rsig.c revent.c rsketch.c rindex.c lchain.c rseed.c rmap.c \
       dtw.c hit.c rextdata.c main.c

OBJS = $(SRCS:.c=.o)

PROG = rawhash2

# =====================================================================
# Determine which external dependencies need building
# =====================================================================
EXT_TARGETS =

NEED_ZSTD = 0
ifneq ($(NOPOD5),1)
    NEED_ZSTD = 1
endif
ifneq ($(NOSLOW5),1)
    NEED_ZSTD = 1
endif
ifeq ($(NEED_ZSTD),1)
    EXT_TARGETS += zstd
endif

ifneq ($(NOHDF5),1)
    EXT_TARGETS += hdf5
endif

ifneq ($(NOSLOW5),1)
    EXT_TARGETS += slow5
endif

ifneq ($(NOPOD5),1)
    EXT_TARGETS += pod5
endif

# Validation targets (run after deps are built)
CHECK_TARGETS =
ifneq ($(NOPOD5),1)
    CHECK_TARGETS += check_pod5
endif
ifneq ($(NOHDF5),1)
    CHECK_TARGETS += check_hdf5
endif
ifneq ($(NOSLOW5),1)
    CHECK_TARGETS += check_slow5
endif

# =====================================================================
# Top-level targets
# =====================================================================
.PHONY: all subset clean help zstd hdf5 slow5 pod5 \
        check_hdf5 check_slow5 check_pod5

all: $(EXT_TARGETS) $(CHECK_TARGETS) $(PROG)

subset: $(CHECK_TARGETS) $(PROG)

# =====================================================================
# Link
# =====================================================================
$(PROG): $(OBJS)
	$(COMPILER) $(ALL_FLAGS) $(OBJS) -o $@ $(LIBS)

# =====================================================================
# Compile: all .c files use the same compiler (CC or CXX)
# =====================================================================
%.o: %.c
	$(COMPILER) -c $(LANG_MODE) $(ALL_FLAGS) $(INCLUDES) $< -o $@

# =====================================================================
# External dependency: zstd
# =====================================================================
zstd:
ifeq ($(NEED_ZSTD),1)
	@echo "=== Building zstd ==="
	+$(MAKE) -C $(WORKDIR)/zstd lib-release
endif

# =====================================================================
# External dependency: HDF5
# =====================================================================
hdf5:
ifneq ($(NOHDF5),1)
ifneq ($(HDF5_COMPILE),0)
	@echo "=== Building HDF5 ==="
	cd $(HDF5_DIR) && \
		mkdir -p build && \
		./configure --enable-threadsafe --disable-hl --prefix="$(HDF5_BUILD_DIR)" && \
		$(MAKE) -j && \
		$(MAKE) install
endif
endif

# =====================================================================
# External dependency: slow5lib (depends on zstd)
# =====================================================================
slow5: zstd
ifneq ($(NOSLOW5),1)
ifneq ($(SLOW5_COMPILE),0)
	@echo "=== Building slow5lib ==="
	+$(MAKE) -C $(SLOW5_DIR) \
		slow5_mt=1 \
		zstd_local=$(WORKDIR)/zstd/lib \
		lib/libslow5.a
endif
endif

# =====================================================================
# External dependency: POD5 (download pre-built, depends on zstd)
# =====================================================================
pod5: zstd
ifneq ($(NOPOD5),1)
ifneq ($(POD5_DOWNLOAD),0)
	@echo "=== Downloading POD5 v$(POD5_VERSION) ==="
	@mkdir -p $(POD5_DIR)
	@if [ ! -f "$(POD5_INCLUDE_DIR)/pod5_format/c_api.h" ]; then \
		echo "Fetching $(POD5_URL)"; \
		curl -fSL $(POD5_URL) | tar -xz -C $(POD5_DIR); \
	else \
		echo "POD5 already downloaded, skipping."; \
	fi
endif
endif

# =====================================================================
# Validation checks
# =====================================================================
check_pod5:
ifneq ($(NOPOD5),1)
	@[ -f "$(POD5_INCLUDE_DIR)/pod5_format/c_api.h" ] || \
		{ echo "ERROR: POD5 headers not found at $(POD5_INCLUDE_DIR)" >&2; exit 1; }
	@[ -f "$(POD5_LIB_DIR)/libpod5_format.a" ] || \
		{ echo "ERROR: POD5 library not found at $(POD5_LIB_DIR)" >&2; exit 1; }
endif

check_hdf5:
ifneq ($(NOHDF5),1)
	@[ -f "$(HDF5_INCLUDE_DIR)/H5pubconf.h" ] || \
		{ echo "ERROR: HDF5 headers not found at $(HDF5_INCLUDE_DIR)" >&2; exit 1; }
	@[ -f "$(HDF5_LIB_DIR)/lib$(HDF5_LIB).a" ] || \
		[ -f "$(HDF5_LIB_DIR)/lib$(HDF5_LIB).so" ] || \
		{ echo "ERROR: HDF5 library not found at $(HDF5_LIB_DIR)" >&2; exit 1; }
endif

check_slow5:
ifneq ($(NOSLOW5),1)
	@[ -f "$(SLOW5_INCLUDE_DIR)/slow5/slow5.h" ] || \
		{ echo "ERROR: SLOW5 headers not found at $(SLOW5_INCLUDE_DIR)" >&2; exit 1; }
	@[ -f "$(SLOW5_LIB_DIR)/libslow5.a" ] || \
		{ echo "ERROR: SLOW5 library not found at $(SLOW5_LIB_DIR)" >&2; exit 1; }
endif

# =====================================================================
# Clean
# =====================================================================
clean:
	rm -f *.o $(PROG) *~

# =====================================================================
# Help
# =====================================================================
help:
	@echo ""
	@echo "src/Makefile options:"
	@echo "  DEBUG=1       Debug build (-O2 -fsanitize=address -g)"
	@echo "  PROFILE=1     Profiling build (-g -fno-omit-frame-pointer)"
	@echo "  NOPOD5=1      Disable POD5 support"
	@echo "  NOHDF5=0      Enable HDF5/FAST5 support (disabled by default)"
	@echo "  NOSLOW5=0     Enable SLOW5/BLOW5 support (disabled by default)"
	@echo "  asan=1        Enable AddressSanitizer"
	@echo "  tsan=1        Enable ThreadSanitizer"
	@echo ""
	@echo "  make subset   Skip recompiling external dependencies"

# =====================================================================
# Header dependency rules
# =====================================================================
rsketch.o: rutils.h rh_kvec.h
lchain.o: kalloc.h rutils.h rseed.h rsketch.h chain.h krmq.h
hit.o: chain.h kalloc.h khash.h rmap.h
rsig.o: rsig.h rh_kvec.h
rseed.o: rsketch.h kalloc.h rutils.h rindex.h
rmap.o: rmap.h rindex.h rsig.h kthread.h rh_kvec.h rutils.h rsketch.h revent.h sequence_until.h dtw.h chain.h khash.h
revent.o: roptions.h kalloc.h
rindex.o: rindex.h roptions.h rutils.h rsketch.h rsig.h bseq.h khash.h rh_kvec.h kthread.h revent.h
main.o: rawhash.h ketopt.h rutils.h
dtw.o: dtw.h rutils.h
