cmake_minimum_required(VERSION 3.0)
project(nekBench VERSION 0.1)

enable_language(C)
enable_language(CXX)
enable_language(Fortran)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(FortranCInterface)

set(dfloat "double")
set(dfloatString "\"double\"")
set(dfloatFormat "\"%lf\"")
set(dlong "long int")
set(dlongString "\"long int\"")
set(dlongFormat="\"%ld\"")
set(hlong "long long int")
set(hlongString "\"long long int\"")
set(hlongFormat "\"%lld\"")
set(MPI_DFLOAT "MPI_DOUBLE")
set(MPI_DLONG "MPI_LONG_INT")
set(MPI_HLONG "MPI_LONG_LONG_INT")

option(USE_MPI "Use the MPI library." ON)
if(USE_MPI)
    find_package(MPI)
endif()

option(EXTERNAL_BLASLAPACK "Use an external/vendor-provided Blas/Lapack library" OFF)
if(EXTERNAL_BLASLAPACK)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

option(EXTERNAL_OCCA "Use an external OCCA build" OFF)
if(EXTERNAL_OCCA)
  find_package(occa REQUIRED)
endif()

add_compile_definitions(
    "USE_OCCA_MEM_BYTE_ALIGN=64"
    "dfloat=double"
    "dfloatString=\"double\""
    "dfloatFormat=\"%lf\""
    "MPI_DFLOAT=MPI_DOUBLE"
    "dlong=int"
    "dlongString=\"int\""
    "dlongFormat=\"%d\""
    "MPI_DLONG=MPI_INT"
    "hlong=long long int"
    "hlongString=\"long long int\""
    "hlongFormat=\"%lld\""
    "MPI_HLONG=MPI_LONG_LONG_INT"
) 

add_subdirectory(3rdParty)
add_subdirectory(axhelm)
#add_subdirectory(gs)
add_subdirectory(nekBone)

configure_file(
  ${CMAKE_SOURCE_DIR}/modulefiles/nekbench
  ${CMAKE_INSTALL_PREFIX}/.modulefiles/nekbench
)