#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>

// Function to calculate batch accuracy
int calculate_batch_accuracy(const std::vector<std::vector<float>>& predictions,
                             const std::vector<std::vector<float>>& targets);

namespace Utils {
    // Generate a random float between min and max
    float randomFloat(float min, float max);

    // Print a 1D vector (for debugging)
    void printVector(const std::vector<float>& vec, const std::string& label = "");

    // Print a 2D vector (for debugging)
    void printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& label = "");

    // Initialize a 2D vector with random values
    std::vector<std::vector<float>> initializeRandomMatrix(size_t rows, size_t cols, float min = -0.1f, float max = 0.1f);

    // Initialize a 1D vector with random values
    std::vector<float> initializeRandomVector(size_t size, float min = -0.1f, float max = 0.1f);

    // Normalize a vector (min-max scaling)
    std::vector<float> normalize(const std::vector<float>& vec);

    // Apply softmax to a vector
    std::vector<float> softmax(const std::vector<float>& vec);
    
}

#endif // UTILS_HPP
