#ifndef CUDA_MANDELBROT_H
#define CUDA_MANDELBROT_H

// CUDA kernel declaration
__global__ void mandelbrotKernel(int *image, float minX, float maxX, float minY, float maxY, int width, int height, int max_iter);

// CUDA wrapper function declaration
void MandelbrotKernel_wrapper(int* d_image, int width, int height, int max_iter);

#endif // CUDA_MANDELBROT_H
