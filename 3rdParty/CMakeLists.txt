#if(NOT EXTERNAL_BLASLAPACK)
#  add_subdirectory(BlasLapack)
#endif()

if(NOT ${EXTERNAL_OCCA})
  add_subdirectory(occa)
endif()

add_subdirectory(gslib)
add_subdirectory(ogs)