#include "cpu_vector_max.h"
#include "timer.h"
#include <random>


// CPU max function
float cpu_max(const std::vector<float>& v) {
    auto& timer = util::timers.cpu_add("CPU Max");
    
    float max_val = v[0];
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > max_val) {
            max_val = v[i];
        }
    }

    timer.stop();
    return max_val;
}
