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

.PHONY: target
target: amgel.so

amgel.so: amgel_dynamic.so
	ln -sf $(CURDIR)/$< amgel.so

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

amgel_static.so: amgel.cpp $(FRIDA_CORE_A) $(FRIDA_GUM_A)
	nvcc -Wno-deprecated-gpu-targets -shared -Xcompiler -fPIC -std=c++11 -lmpi -lnccl -lelf -rdc=true --cudart=shared -I$(FRIDA_DIR) -L$(FRIDA_DIR) -I./atlc/include ./amgel.cpp -lfrida-core -o amgel_static.so
	ln -sf $(CURDIR)/amgel_static.so amgel.so

amgel_dynamic.so: amgel.cpp $(FRIDA_CORE_SO) $(FRIDA_GUM_SO)
	nvcc -Wno-deprecated-gpu-targets -shared -Xcompiler -fPIC -std=c++11 -lmpi -lnccl -lelf -rdc=true --cudart=shared -I$(FRIDA_DIR) -L$(FRIDA_SO) -Xlinker -rpath,$(CURDIR)/$(FRIDA_SO) -I./atlc/include ./amgel.cpp -lfrida-core -o amgel_dynamic.so
	ln -sf $(CURDIR)/amgel_dynamic.so amgel.so

.PHONY: clean
clean:
	$(RM) amgel.so amgel_dynamic.so amgel_static.so
