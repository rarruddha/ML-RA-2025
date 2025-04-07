#include "image_utils.h"
#include <fstream>  // For file handling
#include <iostream>


// Function to save the image in PPM format
void saveImage(int *h_image, int width, int height) {
    // Open a file to write the image
    std::ofstream outFile("mandelbrot.ppm");

    if (!outFile) {
        std::cerr << "Error opening file for writing" << std::endl;
        return;
    }

    // Write the PPM header
    outFile << "P3\n";  // P3 for ASCII format (P6 would be for binary)
    outFile << width << " " << height << "\n";  // Image dimensions
    outFile << "255\n";  // Maximum color value (255 for 8-bit colors)

    // Write the pixel data
    for (int i = 0; i < width * height; ++i) {
        int pixel = h_image[i];

        // Extract RGB components from the packed integer
        int r = (pixel >> 16) & 0xFF;  // Extract red component
        int g = (pixel >> 8) & 0xFF;   // Extract green component
        int b = pixel & 0xFF;          // Extract blue component

        // Write the RGB values to the file
        outFile << r << " " << g << " " << b << "\n";
    }

    // Close the file
    outFile.close();
    std::cout << "Image saved as mandelbrot.ppm" << std::endl;
}