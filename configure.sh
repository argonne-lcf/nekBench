#!/bin/bash
set -x
#-----
export CRAYPE_LINK_TYPE=dynamic
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NVIDIA_PATH}/cuda/lib64:${NVIDIA_PATH}/compilers/lib

BUILD_TYPE=Release
OCCA_ROOT="/home/krowe/occa/install"
PREFIX_PATHS="${NVIDIA_PATH}/cuda;${NVIDIA_PATH}/compilers/lib;${OCCA_ROOT}"

MPICC=`which cc`
MPIFC=`which ftn`

CXX=`which CC`
CC=`which cc`
FC=`which ftn`

# CXXFLAGS="-pgf90libs"
ENABLE_OPENMP="OFF"

# Default build parameters
: ${BUILD_DIR:=`pwd`/build}
: ${INSTALL_DIR:=`pwd`/install}
: ${BUILD_TYPE:="RelWithDebInfo"}

: ${CC:="gcc"}
: ${CXX:="g++"}
: ${FC:="gfortran"}

: ${MPICC:="mpicc"}
: ${MPIFC:="mpif77"}

: ${EXTERNAL_BLASLAPACK:="ON"}

rm -rf ${BUILD_DIR} ${INSTALL_DIR}

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
  -DEXTERNAL_BLASLAPACK=${EXTERNAL_BLASLAPACK}

#cmake --build ${BUILD_DIR} --parallel 32 && \
#cmake --install ${BUILD_DIR} --prefix ${INSTALL_DIR}
