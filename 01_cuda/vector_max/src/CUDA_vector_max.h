#ifndef CUDA_VECTOR_MAX_H
#define CUDA_VECTOR_MAX_H
#include <vector> 

#ifdef __cplusplus
extern "C" {
#endif

void kernel_wrapper(const std::vector<float>& v, int block_size);

#ifdef __cplusplus
}
#endif

#endif // CUDA_VECTOR_MAX_H