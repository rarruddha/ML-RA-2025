#include "cpu_mandelbrot.h"
#include <cmath>
#include "timer.h"  

void mandelbrotCPU(int *image, float minX, float maxX, float minY, float maxY, int width, int height, int max_iter) {
    auto& timer = util::timers.cpu_add("CPU Mandelbrot");  // Start CPU timer

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
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

            // Storing RGB values in the image array
            image[y * width + x] = (r << 16) | (g << 8) | b;
        }
    }
    timer.stop();  
}
