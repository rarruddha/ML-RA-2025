enable_language(CUDA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set_source_files_properties(CUDA_vector_max.cu PROPERTIES LANGUAGE CUDA)

add_executable(vector_max
    main.cpp
    cpu_vector_max.cpp
    cpu_vector_max.h
    CUDA_vector_max.cu
    CUDA_vector_max.h
    error.cpp
    error.h
    timer.cpp
    timer.h
)

set_target_properties(vector_max PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)