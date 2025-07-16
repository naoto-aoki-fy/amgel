FRIDA_VERSION = 17.2.6
CPUARCH = $(shell uname -m)
KERNEL = $(shell uname -s | tr 'A-Z' 'a-z')
FRIDA_POSTFIX = $(FRIDA_VERSION)-$(KERNEL)-${CPUARCH}.tar.xz
FRIDA_CORE_TARXZ = frida-core-devkit-$(FRIDA_POSTFIX)
FRIDA_GUM_TARXZ = frida-gum-devkit-$(FRIDA_POSTFIX)
FRIDA_CORE_A = frida/libfrida-core.a
FRIDA_GUM_A = frida/libfrida-gum.a

.Phony: target
target: amgel.so

.Phony: frida
frida: $(FRIDA_CORE_A) $(FRIDA_GUM_A)

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

$(FRIDA_GUM_TARXZ):
	curl -LO https://github.com/frida/frida/releases/download/$(FRIDA_VERSION)/$(FRIDA_GUM_TARXZ)
	# touch -t 197001020001.00 $(FRIDA_GUM_TARXZ)

amgel.so: amgel.cpp frida
	nvcc -Wno-deprecated-gpu-targets -shared -Xcompiler -fPIC -std=c++11 -lmpi -lnccl -lelf -rdc=true --cudart=shared -I./frida -L./frida -I./atlc/include ./amgel.cpp -lfrida-core -o amgel.so

.Phony: clean
clean:
	$(RM) amgel.so