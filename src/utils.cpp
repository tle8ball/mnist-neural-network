#include "../include/utils.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>


int calculate_batch_accuracy(const std::vector<std::vector<float>>& predictions,
                             const std::vector<std::vector<float>>& targets) {
    int correct = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        // Find the index of the maximum value in predictions (predicted class)
        auto pred_class = std::distance(predictions[i].begin(),
                                        std::max_element(predictions[i].begin(), predictions[i].end()));

        // Find the index of the maximum value in targets (true class)
        auto true_class = std::distance(targets[i].begin(),
                                        std::max_element(targets[i].begin(), targets[i].end()));

        if (pred_class == true_class) {
            ++correct;
        }
    }

    return correct;
}

// Generate a random float between min and max
float Utils::randomFloat(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Print a 1D vector (for debugging)
void Utils::printVector(const std::vector<float>& vec, const std::string& label) {
    if (!label.empty()) {
        std::cout << label << ": ";
    }
    for (float val : vec) {
        std::cout << std::fixed << std::setprecision(3) << val << " ";
    }
    std::cout << std::endl;
}

// Print a 2D vector (for debugging)
void Utils::printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& label) {
    if (!label.empty()) {
        std::cout << label << ":\n";
    }
    for (const auto& row : matrix) {
        Utils::printVector(row);
    }
}

// Initialize a 2D vector with random values
std::vector<std::vector<float>> Utils::initializeRandomMatrix(size_t rows, size_t cols, float min, float max) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = Utils::randomFloat(min, max);
        }
    }
    return matrix;
}

// Initialize a 1D vector with random values
std::vector<float> Utils::initializeRandomVector(size_t size, float min, float max) {
    std::vector<float> vec(size);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = Utils::randomFloat(min, max);
    }
    return vec;
}

// Normalize a vector (min-max scaling)
std::vector<float> Utils::normalize(const std::vector<float>& vec) {
    float min_val = *std::min_element(vec.begin(), vec.end());
    float max_val = *std::max_element(vec.begin(), vec.end());
    std::vector<float> normalized(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        normalized[i] = (vec[i] - min_val) / (max_val - min_val);
    }
    return normalized;
}

// Apply softmax to a vector
std::vector<float> Utils::softmax(const std::vector<float>& vec) {
    std::vector<float> exp_vec(vec.size());
    float sum_exp = 0.0f;

    // Compute exponential values and their sum
    for (size_t i = 0; i < vec.size(); ++i) {
        exp_vec[i] = std::exp(vec[i]);
        sum_exp += exp_vec[i];
    }

    // Normalize by dividing each exponential value by the sum
    for (size_t i = 0; i < exp_vec.size(); ++i) {
        exp_vec[i] /= sum_exp;
    }

    return exp_vec;
}
