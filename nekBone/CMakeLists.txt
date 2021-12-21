project(nekbone)

add_executable(nekBone
  main.cpp 
  setup.cpp
  solve.cpp
  ${CMAKE_SOURCE_DIR}/core/timer.cpp
  ${CMAKE_SOURCE_DIR}/core/meshSetup.cpp
  ${CMAKE_SOURCE_DIR}/core/meshBasis.cpp
  ${CMAKE_SOURCE_DIR}/core/setupAide.cpp
)

target_include_directories(nekBone 
  PUBLIC 
    ${CMAKE_SOURCE_DIR}/core 
    ${CMAKE_SOURCE_DIR}/axhelm
)

target_link_libraries(nekBone 
  PUBLIC
    MPI::MPI_C
    OCCA::libocca
    ogs
    BLAS::BLAS
    LAPACK::LAPACK
)

add_compile_definitions(nekBone PUBLIC "DBP=\"\"")

install(TARGETS nekBone RUNTIME DESTINATION ".")
install(DIRECTORY kernel/ DESTINATION kernel)
install(FILES setup.ini DESTINATION "." RENAME nekBone.ini)
