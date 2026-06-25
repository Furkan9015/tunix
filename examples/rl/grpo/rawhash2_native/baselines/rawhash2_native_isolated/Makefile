# RawHash2-compatible isolated baseline.
#
# This root makefile intentionally exposes only the standalone source build used
# by the RLVR benchmark. It delegates to src/Makefile and copies the resulting
# binary to bin/rawhash2, matching the runner contract.

.PHONY: all rawhash2 subset clean help

all: rawhash2

rawhash2:
	@mkdir -p bin
	+$(MAKE) -C src
	@cp src/rawhash2 bin/rawhash2
	@echo "Build complete: bin/rawhash2"

subset: rawhash2

clean:
	rm -rf bin
	+$(MAKE) -C src clean

help:
	@echo "RawHash2-compatible isolated baseline"
	@echo ""
	@echo "Targets:"
	@echo "  make                  Build bin/rawhash2"
	@echo "  make clean            Remove build artifacts"
	@echo ""
	@echo "Benchmark build flags:"
	@echo "  PROFILE=1 NOPOD5=1 NOHDF5=0 NOSLOW5=1"
	@echo "  HDF5_INCLUDE_DIR=/path/to/hdf5/include"
	@echo "  HDF5_LIB_DIR=/path/to/hdf5/lib"
