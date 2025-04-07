enable_language(CUDA)

add_executable(mandelbrot_set
    main.cpp
    cpu_mandelbrot.cpp
    cpu_mandelbrot.h
    CUDA_mandelbrot.cu
    CUDA_mandelbrot.h
    error.cpp
    error.h
    timer.cpp
    timer.h
    image_utils.cpp
    image_utils.h
)

target_link_libraries(mandelbrot_set ${CUDA_LIBRARIES})

set_target_properties(mandelbrot_set PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

