#include "../include/optimizer.hpp"
#include <cstddef>

void SGDOptimizer::update(std::vector<std::vector<float>>& weights, const std::vector<std::vector<float>>& gradients) {
    for (std::size_t i = 0; i < weights.size(); ++i) {
        for (std::size_t j = 0; j < weights[i].size(); ++j) {
            float grad = gradients[i][j];
            // Clip gradients to range [-1.0, 1.0]
            if (grad > 1.0f) grad = 1.0f;
            if (grad < -1.0f) grad = -1.0f;
            weights[i][j] -= learning_rate * grad;
        }
    }
}

void SGDOptimizer::update(std::vector<float>& biases, const std::vector<float>& gradients) {
    for (std::size_t i = 0; i < biases.size(); ++i) {
        float grad = gradients[i];
        // Clip gradients to range [-1.0, 1.0]
        if (grad > 1.0f) grad = 1.0f;
        if (grad < -1.0f) grad = -1.0f;
        biases[i] -= learning_rate * grad;
    }
}