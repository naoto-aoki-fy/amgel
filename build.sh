#!/bin/sh

set -e
set -x

if [[ ./cdl86/cdl.c -nt cdl.o ]]; then
  gcc -c -fPIC -I./cdl86 ./cdl86/cdl.c -o ./cdl.o
fi

if [[ ./mynccl.cpp -nt ./mynccl.so ]]; then
  nvcc -Wno-deprecated-gpu-targets -shared -Xcompiler -fPIC -std=c++11 -lmpi -lnccl -lelf -rdc=true --cudart=shared -I./cdl86 -I./atlc/include ./mynccl.cpp cdl.o -o mynccl.so
fi