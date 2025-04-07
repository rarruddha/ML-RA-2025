#include <iostream>
#include <unistd.h>
#include <vector>
#include <random>

#include "timer.h"
#include "CUDA_vector_max.h"
#include "cpu_vector_max.h"
#include "error.h"

using namespace std;

// Generating my random vector v
std::vector<float> create_random_vector(size_t size) {
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(size);
    for (size_t i = 0; i < size; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

int main(int argc, const char* argv[]) {
    cout << "Launching Parallel Max CUDA..." << endl;

    // Default parameters
    size_t vector_size = 1 << 20;  // default: 2^20 = 1048576
    int block_size = 32;           // default: 32 threads per block

    // Command line arguments to run different tests
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg.rfind("size:", 0) == 0) {
            vector_size = std::stoul(arg.substr(5));
        } else if (arg.rfind("blocksize:", 0) == 0) {
            block_size = std::stoi(arg.substr(10));
        }
    }

    std::vector<float> v = create_random_vector(vector_size);
    cout << "Vector size: " << v.size() << ", Block size: " << block_size << endl;

    // CPU version
    float max_cpu = cpu_max(v);
    cout << "CPU Max value: " << max_cpu << endl;

    // CUDA kernel
    kernel_wrapper(v,block_size);

    util::timers.flush(); //

    // sleep some seconds, otherwise 
    // device printf won't appear in the terminal
    cout << "sleeping for a few seconds..." << endl;
    sleep(10);

    return 0;
}