#include "CUDA_mandelbrot.h"
#include <cmath>
#include "timer.h"  
#include "error.h"  

// CUDA kernel to compute Mandelbrot set
__global__ void mandelbrotKernel(int *image, float minX, float maxX, float minY, float maxY, int width, int height, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Map pixel coordinates to complex plane
        float real = minX + (maxX - minX) * x / width;
        float imag = minY + (maxY - minY) * y / height;

        float cReal = real;
        float cImag = imag;
        float zReal = 0.0f;
        float zImag = 0.0f;
        int iterations = 0;

        // Mandelbrot iteration
        while (zReal * zReal + zImag * zImag <= 4.0f && iterations < max_iter) {
            float tempReal = zReal * zReal - zImag * zImag + cReal;
            zImag = 2.0f * zReal * zImag + cImag;
            zReal = tempReal;
            iterations++;
        }

        // RGB coloring
        int r, g, b;
        if (iterations == max_iter) {
            r = 0;
            g = 0;
            b = 0;  // Inside Mandelbrot set (black); ie these are the points that do not escape to infinity
        } else {
            // Here we chose a color gradient based on the number of iterations. 
            //Recall that 256 is the max value of a byte.
            //For the values that escape, we use the following color mapping:

            r = (iterations * 15) % 256;
            g = (iterations * 30) % 256;
            b = (iterations * 45) % 256;

            // Ie, points that escape quickly will have low values of r,g,b (be closer to black)
            // and points that escape slowly will have high values of r,g,b (be closer to white)
        }

        // Storing the RGB values in the image array
        image[y * width + x] = (r << 16) | (g << 8) | b;  
    }
}

void MandelbrotKernel_wrapper(int* d_image, int width, int height, int max_iter) {
    auto& timer = util::timers.gpu_add("CUDA Mandelbrot"); // Start GPU timer

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mandelbrotKernel<<<numBlocks, threadsPerBlock>>>(d_image, -2.0f, 1.0f, -1.5f, 1.5f, width, height, max_iter);

   
    util::check_cuda_error("CUDA kernel launch failed");

    
    cudaDeviceSynchronize();
    timer.stop();
}
