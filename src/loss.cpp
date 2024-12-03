#include "../include/loss.hpp"

float CrossEntropyLoss::calculate_loss(const std::vector<std::vector<float>>& predictions,
                                      const std::vector<std::vector<float>>& targets) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < predictions.size(); ++i) {
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            // Cross-Entropy: -sum(target * log(prediction))
            total_loss += targets[i][j] * std::log(predictions[i][j] + 1e-9f); // Add epsilon to avoid log(0)
        }
    }

    return -total_loss / static_cast<float>(predictions.size()); // Average loss
}

std::vector<std::vector<float>> CrossEntropyLoss::calculate_gradient(const std::vector<std::vector<float>>& predictions,
                                                                       const std::vector<std::vector<float>>& targets) {
    std::vector<std::vector<float>> gradients = predictions;

    for (size_t i = 0; i < predictions.size(); ++i) {
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            gradients[i][j] -= targets[i][j]; // Gradient: predictions - targets
        }
    }

    return gradients;
}