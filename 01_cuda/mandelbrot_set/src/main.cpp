#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cstdlib>  // For std::atoi
#include <cstring>  // For std::strtok
#include "CUDA_mandelbrot.h"
#include "timer.h"  
#include "image_utils.h"
#include "cpu_mandelbrot.h"


int main(int argc, char *argv[]) {
    // Default values
    int max_iter = 1000; 
    int width = 800;     // (800 pixels wide)
    int height = 800;    // (800 pixels high)

    // Default window for complex plane. I found these to be the standard values. 
    float minX = -2.0f;  
    float maxX = 1.0f;   
    float minY = -1.5f;  
    float maxY = 1.5f;   

    // The following code is used to parse command line arguments
    // this ensures that we can run the program with different parameters of max_iterations, width and height.
    // For instance we can run ./mandelbrot_set max_iter:1000 width:800 height:800
    // and the program will use these values instead of the default ones defined above.
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "max_iter:", 9) == 0) {
            max_iter = std::atoi(argv[i] + 9);  

        } else if (strncmp(argv[i], "width:", 6) == 0) {
            width = std::atoi(argv[i] + 6);  

        } else if (strncmp(argv[i], "height:", 7) == 0) {
            height = std::atoi(argv[i] + 7);  
        }
    }


    std::cout << "Using values: max_iter=" << max_iter << ", width=" << width << ", height=" << height << std::endl;

    // Allocating memory for the image 
    int *d_image, *h_image;
    h_image = (int *)malloc(width * height * sizeof(int));
    cudaMalloc((void **)&d_image, width * height * sizeof(int));

    // CPU Mandelbrot computation 
    mandelbrotCPU(h_image, minX, maxX, minY, maxY, width, height, max_iter);

    // Launch CUDA kernel
    MandelbrotKernel_wrapper(d_image, width, height, max_iter);  // For GPU

    // Copying the results back to host memory
    cudaMemcpy(h_image, d_image, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Saving the image in ppm format
    saveImage(h_image, width, height);

    // Free allocated memory
    cudaFree(d_image);
    free(h_image);

    // Flush the timer results after execution
    util::timers.flush();  // Ensure timers output their data

    std::cout << "Mandelbrot image saved as mandelbrot.ppm" << std::endl;

    return 0;
}
