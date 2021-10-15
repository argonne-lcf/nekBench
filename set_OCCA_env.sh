#!/bin/bash

module load cmake
module use /module/files/on/jlse 
module load oneapi

#OCCA_DIR=/path/to/occa

export PATH+=":${OCCA_DIR}/bin"
export LD_LIBRARY_PATH+=":${OCCA_DIR}/lib"
export SYCL_DEVICE_FILTER=gpu
export OCCA_CXX="icpx"
export OCCA_CXXFLAGS="-O3"
export OCCA_DPCPP_COMPILER="dpcpp"
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device 0x020a\""
