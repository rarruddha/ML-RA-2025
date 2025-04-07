#ifndef CPU_VECTOR_MAX_H
#define CPU_VECTOR_MAX_H

#include <vector>

// Vector of normally distributed random floats
std::vector<float> create_random_vector(size_t size);

// Max val using CPU
float cpu_max(const std::vector<float>& v);

#endif // CPU_VECTOR_MAX_H