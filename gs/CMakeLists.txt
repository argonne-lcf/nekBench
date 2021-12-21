project(gs-bench)

add_executable(gs-bench
  main.cpp
  ping_pong.cpp
  mygs.cpp
  ${CMAKE_SOURCE_DIR}/core/timer.cpp
  ${CMAKE_SOURCE_DIR}/core/meshSetup.cpp
  ${CMAKE_SOURCE_DIR}/core/meshBasis.cpp
  ${CMAKE_SOURCE_DIR}/core/setupAide.cpp
)

target_include_directories(gs-bench PUBLIC ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(gs-bench 
  PUBLIC
    MPI::MPI_C 
    OCCA::libocca
    ogs
    BLAS::BLAS
    LAPACK::LAPACK
)

install(TARGETS gs-bench RUNTIME DESTINATION ".")
install(FILES gs_dummy.okl DESTINATION ".")