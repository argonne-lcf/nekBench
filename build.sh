#!/bin/bash
set -x
#-----
PREFIX_PATHS=

# Default build parameters
: ${BUILD_DIR:=`pwd`/build}
: ${INSTALL_DIR:=`pwd`/install}
: ${BUILD_TYPE:="RelWithDebInfo"}

: ${CC:="gcc"}
: ${CXX:="g++"}
: ${FC:="gfortran"}

: ${MPICC:="mpicc"}
: ${MPICXX:="mpicxx"}
: ${MPIFC:="mpif77"}

: ${VENDOR_BLASLAPACK:="OFF"}

# OCCA Configuration
: ${ENABLE_DPCPP:="ON"}
: ${ENABLE_OPENCL:="ON"}
: ${ENABLE_CUDA:="OFF"}
: ${ENABLE_HIP="OFF"}
: ${ENABLE_OPENMP="OFF"}
: ${ENABLE_METAL="OFF"}
: ${ENABLE_MPI="OFF"}

cmake -S . -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_PREFIX_PATH=${PREFIX_PATHS} \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DCMAKE_Fortran_COMPILER=${FC} \
  -DMPI_C_COMPILER=${MPICC} \
  -DMPI_CXX_COMPILER=${MPICXX} \
  -DMPI_Fortran_COMPILER=${MPIFC} \
  -DVENDOR_BLASLAPACK=${VENDOR_BLASLAPACK} \
  -DENABLE_DPCPP=${ENABLE_DPCPP} \
  -DENABLE_OPENCL=${ENABLE_OPENCL} \
  -DENABLE_CUDA=${ENABLE_CUDA} \
  -DENABLE_HIP=${ENABLE_HIP} \
  -DENABLE_OPENMP=${ENABLE_OPENMP} \
  -DENABLE_METAL=${ENABLE_METAL} \
  -DENABLE_MPI=${ENABLE_MPI}

cmake --build ${BUILD_DIR} --parallel 4 && \
cmake --install ${BUILD_DIR} --prefix ${INSTALL_DIR}

# mkdir -p ${INSTALL_DIR}
# cmake --install build/3rdParty/occa --prefix ${INSTALL_DIR}/occa
# cmake --install build/axhelm --prefix ${INSTALL_DIR}
# cmake --install build/nekBone --prefix ${INSTALL_DIR}