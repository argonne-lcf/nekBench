#!/bin/bash

module load cmake
module use /soft/restricted/intel_dga/modulefiles 
module load oneapi
#module load oneapi/2021.04.30.004

#export OCCA_DIR=/home/spatel/A21/OCCA/OCCA_DPCPP/date_8.31.21/occa
#export OCCA_DIR=/home/spatel/A21/OCCA/OCCA_DPCPP/occa_9.21/occa/install_OAP_04.30.001/
export OCCA_DIR=/home/spatel/A21/OCCA/OCCA_DPCPP/occa_9.21/occa/install_OAP_08.15.001/
#export OCCA_DIR=/home/spatel/A21/OCCA/OCCA_DPCPP/occa_9.21/occa/install_OAP_08.15.001_RelDebInfo/
#export LIBOMPTARGET_PLUGIN=opencl
#export LIBOMPTARGET_OPENCL_USE_SVM=0

export PATH+=":${OCCA_DIR}/bin"
export LD_LIBRARY_PATH+=":${OCCA_DIR}/lib"
export SYCL_DEVICE_FILTER=gpu
export OCCA_CXX="icpx"
export OCCA_CXXFLAGS="-O3"
export OCCA_DPCPP_COMPILER="dpcpp"
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device 0x020a\""
#export EnableWalkerPartition=0
#export OCCA_CXXFLAGS="-O2"
