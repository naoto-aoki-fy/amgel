-include config.mk

CFLAGS_VENDOR ?=
LDFLAGS_VENDOR ?=
GENCODE_FLAGS ?=
NVCC ?= nvcc --forward-unknown-to-host-compiler
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wno-deprecated-gpu-targets -Xcompiler -fPIC -std=c++11 -rdc=true $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) --cudart=shared
LDLIBS ?= -lmpi -lnccl -lelf

FRIDA_VERSION = 17.2.6
CPUARCH = $(shell uname -m)
KERNEL = $(shell uname -s | tr 'A-Z' 'a-z')
FRIDA_DIR = frida
FRIDA_POSTFIX = $(FRIDA_VERSION)-$(KERNEL)-${CPUARCH}.tar.xz
FRIDA_CORE_TARXZ = frida-core-devkit-$(FRIDA_POSTFIX)
FRIDA_GUM_TARXZ = frida-gum-devkit-$(FRIDA_POSTFIX)
FRIDA_CORE_A = $(FRIDA_DIR)/libfrida-core.a
FRIDA_GUM_A = $(FRIDA_DIR)/libfrida-gum.a
FRIDA_SO = $(FRIDA_DIR)/so
FRIDA_CORE_SO = $(FRIDA_DIR)/so/libfrida-core.so
FRIDA_GUM_SO = $(FRIDA_DIR)/so/libfrida-gum.so

.PHONY: target test semantic-test differential differential-run
target: ncclfold.so

DIFFERENTIAL_BIN ?= tests/differential/nccl_semantics_test
SEMANTIC_BIN ?= tests/semantic/nccl_semantics_test

differential: $(DIFFERENTIAL_BIN)

$(DIFFERENTIAL_BIN): tests/differential/nccl_semantics_test.cu
	$(NVCC) $(CFLAGS_VENDOR) -std=c++14 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS) $< \
		$(LDFLAGS_VENDOR) --cudart=shared -lmpi -lnccl -o $@

$(SEMANTIC_BIN): tests/differential/nccl_semantics_test.cu
	$(NVCC) $(CFLAGS_VENDOR) -std=c++14 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS) $< \
		$(LDFLAGS_VENDOR) --cudart=shared -lmpi -lnccl -o $@

differential-run: differential ncclfold.so
	python3 tests/differential/run_differential.py --ncclfold ./ncclfold.so

# The public, single-GPU correctness suite.  TEST_RANKS and TEST_ARGS make it
# easy for CI and local installations to select rank counts or launcher flags.
TEST_RANKS ?= 2,4
TEST_ARGS ?=
semantic-test: $(SEMANTIC_BIN) ncclfold.so
	python3 tests/semantic/run_semantic.py --ranks $(TEST_RANKS) --ncclfold ./ncclfold.so $(TEST_ARGS)

test: semantic-test

ncclfold.so: ncclfold_dynamic.so
	ln -sf $(CURDIR)/$< ncclfold.so

$(FRIDA_CORE_A): $(FRIDA_CORE_TARXZ)
	mkdir -p frida
	tar xvf $(FRIDA_CORE_TARXZ) -C frida
	touch -t "$$(date -d "$$(stat -c %y $(FRIDA_CORE_A)) 1 minute ago" +"%Y%m%d%H%M.%S")" $(FRIDA_CORE_TARXZ)

$(FRIDA_GUM_A): $(FRIDA_GUM_TARXZ)
	mkdir -p frida
	tar xvf $(FRIDA_GUM_TARXZ) -C frida
	touch -t "$$(date -d "$$(stat -c %y $(FRIDA_GUM_A)) 1 minute ago" +"%Y%m%d%H%M.%S")" $(FRIDA_GUM_TARXZ)

$(FRIDA_CORE_TARXZ):
	curl -LO https://github.com/frida/frida/releases/download/$(FRIDA_VERSION)/$(FRIDA_CORE_TARXZ)
	# touch -t 197001020001.00 $(FRIDA_CORE_TARXZ)

$(FRIDA_CORE_SO): $(FRIDA_CORE_A)
	mkdir -p $(FRIDA_DIR)/libfrida-core-o
	mkdir -p $(FRIDA_SO)
	ar x $(FRIDA_CORE_A) --output $(FRIDA_DIR)/libfrida-core-o
	cd $(FRIDA_DIR)/libfrida-core-o && g++ -shared -fPIC *.o .*.o -o ../../$(FRIDA_SO)/libfrida-core.so

$(FRIDA_GUM_TARXZ):
	curl -LO https://github.com/frida/frida/releases/download/$(FRIDA_VERSION)/$(FRIDA_GUM_TARXZ)
	# touch -t 197001020001.00 $(FRIDA_GUM_TARXZ)

$(FRIDA_GUM_SO): $(FRIDA_GUM_A)
	mkdir -p $(FRIDA_DIR)/libfrida-gum-o
	mkdir -p $(FRIDA_SO)
	ar x $(FRIDA_GUM_A) --output $(FRIDA_DIR)/libfrida-gum-o
	cd $(FRIDA_DIR)/libfrida-gum-o && g++ -shared -fPIC *.o .*.o -o ../../$(FRIDA_SO)/libfrida-gum.so

ncclfold_static.so: ncclfold.cpp ncclfold.cu ncclfold.hpp $(FRIDA_CORE_A) $(FRIDA_GUM_A)
	$(NVCC) $(NVCC_CFLAGS) -shared -I$(FRIDA_DIR) ./ncclfold.cpp ./ncclfold.cu $(NVCC_LDFLAGS) -L$(FRIDA_DIR) -lfrida-core $(LDLIBS) -o ncclfold_static.so
	ln -sf $(CURDIR)/ncclfold_static.so ncclfold.so

ncclfold_dynamic.so: ncclfold.cpp ncclfold.cu ncclfold.hpp $(FRIDA_CORE_SO) $(FRIDA_GUM_SO)
	$(NVCC) $(NVCC_CFLAGS) -shared -I$(FRIDA_DIR) ./ncclfold.cpp ./ncclfold.cu $(NVCC_LDFLAGS) -L$(FRIDA_SO) -Xlinker -rpath,$(CURDIR)/$(FRIDA_SO) -lfrida-core $(LDLIBS) -o ncclfold_dynamic.so
	ln -sf $(CURDIR)/ncclfold_dynamic.so ncclfold.so

.PHONY: clean
clean:
	$(RM) ncclfold.so ncclfold_dynamic.so ncclfold_static.so $(DIFFERENTIAL_BIN) $(SEMANTIC_BIN)
