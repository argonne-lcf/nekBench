#!/bin/bash
set -x
#-----
export CRAYPE_LINK_TYPE=dynamic
BUILD_TYPE=Release

EXTERNAL_BLASLAPACK="ON"
EXTERNAL_OCCA="ON"
PREFIX_PATHS="${OCCA_DIR};${NVIDIA_PATH}/cuda"

CC=cc
CXX=CC
FC=ftn

MPICC=cc
MPIFC=ftn

MPICC=cc
MPIFC=ftn

# Default build parameters
: ${BUILD_DIR:=`pwd`/build}
: ${INSTALL_DIR:=`pwd`/install}
: ${BUILD_TYPE:="RelWithDebInfo"}

: ${CC:="gcc"}
: ${CXX:="g++"}
: ${FC:="gfortran"}

: ${MPICC:="mpicc"}
: ${MPIFC:="mpif77"}

: ${EXTERNAL_BLASLAPACK:="OFF"}
: ${EXTERNAL_OCCA:="OFF"}

# OCCA Configuration
: ${ENABLE_DPCPP:="ON"}
: ${ENABLE_OPENCL:="ON"}
: ${ENABLE_CUDA:="ON"}
: ${ENABLE_HIP="ON"}
: ${ENABLE_OPENMP="OFF"}
: ${ENABLE_METAL="OFF"}
: ${ENABLE_MPI="OFF"}

cmake -S . -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DCMAKE_PREFIX_PATH=${PREFIX_PATHS} \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DCMAKE_Fortran_COMPILER=${FC} \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DMPI_C_COMPILER=${MPICC} \
  -DMPI_Fortran_COMPILER=${MPIFC} \
  -DEXTERNAL_BLASLAPACK=${EXTERNAL_BLASLAPACK} \
  -DEXTERNAL_OCCA=${EXTERNAL_OCCA} \
  -DENABLE_DPCPP=${ENABLE_DPCPP} \
  -DENABLE_OPENCL=${ENABLE_OPENCL} \
  -DENABLE_CUDA=${ENABLE_CUDA} \
  -DENABLE_HIP=${ENABLE_HIP} \
  -DENABLE_OPENMP=${ENABLE_OPENMP} \
  -DENABLE_METAL=${ENABLE_METAL} \
  -DENABLE_MPI=${ENABLE_MPI}
