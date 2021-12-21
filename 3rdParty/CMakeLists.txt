#if(NOT VENDOR_BLASLAPACK)
#  add_subdirectory(BlasLapack)
#endif()

#add_subdirectory(occa)
add_subdirectory(gslib)
add_subdirectory(ogs)