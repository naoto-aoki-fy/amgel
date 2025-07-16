#!/bin/sh

set -e
set -x

FRIDA_VERSION=17.2.6

if ! [[ -d frida ]]; then
  CPUARCH="$(uname -m)"
  FRIDA_POSTFIX="${FRIDA_VERSION}-linux-${CPUARCH}.tar.xz"
  FRIDA_CORE_TARXZ="frida-core-devkit-${FRIDA_POSTFIX}"
  FRIDA_GUM_TARXZ="frida-gum-devkit-${FRIDA_POSTFIX}"
  curl -LO "https://github.com/frida/frida/releases/download/${FRIDA_VERSION}/${FRIDA_CORE_TARXZ}"
  curl -LO "https://github.com/frida/frida/releases/download/${FRIDA_VERSION}/${FRIDA_GUM_TARXZ}"
  mkdir frida
  tar xvf "${FRIDA_CORE_TARXZ}" -C frida
  tar xvf "${FRIDA_GUM_TARXZ}" -C frida
fi

if [[ ./amgel.cpp -nt ./amgel.so ]]; then
  nvcc -Wno-deprecated-gpu-targets -shared -Xcompiler -fPIC -std=c++11 -lmpi -lnccl -lelf -rdc=true --cudart=shared -I./frida -L./frida -I./atlc/include ./amgel.cpp -lfrida-core -o amgel.so
fi