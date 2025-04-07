#include <vector>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <random>
#include "timer.h"


typedef std::mt19937 RNG;
using namespace std;

 //Original file had this: (#define BLOCK_SIZE 32)
 // but now we want block_size to vary so that we can test different values
 // This means we also have to change the kernel launch parameters

 __global__ void max_kernel(float *d_v, int n, float *d_max) {
    extern __shared__ float sdata[]; 
    // dynamic shared memory is needed so that we can allocate 
    //the correct amount of memory depending on block size.
    int ti = threadIdx.x;
    int i = blockIdx.x * blockDim.x + ti;

    sdata[ti] = (i < n) ? d_v[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (ti < stride) {
            sdata[ti] = max(sdata[ti], sdata[ti + stride]);
        }
        __syncthreads();
    }

    if (ti == 0) d_max[blockIdx.x] = sdata[0];
}

extern "C" void kernel_wrapper(const std::vector<float>& v, int block_size) {
    cout << "v.size(): " << v.size() << endl;
    cout << "BLOCK_SIZE: " << block_size << endl;

    auto& timer = util::timers.gpu_add("CUDA Max Kernel");

    float *d_v;
    cudaMalloc(&d_v, v.size() * sizeof(float));
    cudaMemcpy(d_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (v.size() + block_size - 1) / block_size;
    float *d_max;
    cudaMalloc(&d_max, numBlocks * sizeof(float));

    dim3 grid(numBlocks);
    dim3 block(block_size);
    size_t shared_mem_size = block_size * sizeof(float);
    max_kernel<<<grid, block, shared_mem_size>>>(d_v, v.size(), d_max);

    vector<float> max_val_host(numBlocks);
    cudaMemcpy(max_val_host.data(), d_max, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float finalMax = max_val_host[0];
    for (int i = 1; i < numBlocks; i++) {
        finalMax = max(finalMax, max_val_host[i]);
    }

    cout << "Maximum value: " << finalMax << endl;

    cudaFree(d_v);
    cudaFree(d_max);
    timer.stop();
}
