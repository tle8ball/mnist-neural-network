#include "../include/optimizer.hpp"
#include <cstddef>

void SGDOptimizer::update(std::vector<std::vector<float>>& weights, const std::vector<std::vector<float>>& gradients) {
    for (std::size_t i = 0; i < weights.size(); ++i) {
        for (std::size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learning_rate * gradients[i][j];
        }
    }
}

void SGDOptimizer::update(std::vector<float>& biases, const std::vector<float>& gradients) {
    for (std::size_t i = 0; i < biases.size(); ++i) {
        biases[i] -= learning_rate * gradients[i];
    }
}