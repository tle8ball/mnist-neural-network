#include "../include/layers.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>
#include <algorithm>

// DenseLayer constructor
DenseLayer::DenseLayer(int input_size, int output_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    weights.resize(input_size, std::vector<float>(output_size));
    biases.resize(output_size, 0.0f);

    for (auto& row : weights) {
        for (float& val : row) {
            val = dist(gen);
        }
    }
}

// DenseLayer forward pass
std::vector<std::vector<float>> DenseLayer::forward(const std::vector<std::vector<float>>& inputs) {
    this->inputs = inputs; // Cache inputs for backpropagation
    std::vector<std::vector<float>> outputs(inputs.size(), std::vector<float>(biases.size(), 0.0f));

    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < biases.size(); ++j) {
            outputs[i][j] = biases[j];
            for (size_t k = 0; k < weights.size(); ++k) {
                outputs[i][j] += inputs[i][k] * weights[k][j];
            }
        }
    }

    return outputs;
}

// DenseLayer backward pass
std::vector<std::vector<float>> DenseLayer::backward(const std::vector<std::vector<float>>& gradient) {
    size_t batch_size = gradient.size();
    size_t input_size = weights.size();
    size_t output_size = weights[0].size();

    // Gradients for weights, biases, and inputs
    weight_gradients.assign(input_size, std::vector<float>(output_size, 0.0f));
    bias_gradients.assign(output_size, 0.0f);
    std::vector<std::vector<float>> input_gradients(batch_size, std::vector<float>(input_size, 0.0f));

    // Calculate weight and bias gradients
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            bias_gradients[j] += gradient[i][j]; // Accumulate bias gradients
            for (size_t k = 0; k < input_size; ++k) {
                weight_gradients[k][j] += inputs[i][k] * gradient[i][j]; // Accumulate weight gradients
                input_gradients[i][k] += gradient[i][j] * weights[k][j]; // Backpropagate to inputs
            }
        }
    }

    // Average gradients over the batch
    for (auto& row : weight_gradients) {
        for (auto& val : row) {
            val /= static_cast<float>(batch_size);
        }
    }
    for (auto& val : bias_gradients) {
        val /= static_cast<float>(batch_size);
    }

    return input_gradients; // Gradient to pass to the previous layer
}


// DenseLayer update
void DenseLayer::update(Optimizer& optimizer) {
    optimizer.update(weights, weight_gradients);
    optimizer.update(biases, bias_gradients);
}

// ActivationLayer implementation
ActivationLayer::ActivationLayer(const std::string& type) : activation_type(type) {}

std::vector<std::vector<float>> ActivationLayer::forward(const std::vector<std::vector<float>>& inputs) {
    this->inputs = inputs; // Cache inputs for backpropagation
    std::vector<std::vector<float>> outputs = inputs;

    if (activation_type == "relu") {
        for (auto& row : outputs) {
            for (auto& val : row) {
                val = std::max(0.0f, val); // ReLU: max(0, x)
            }
        }
    } else if (activation_type == "softmax") {
        for (auto& row : outputs) {
            float max_val = *std::max_element(row.begin(), row.end());
            float sum_exp = 0.0f;
            for (float val : row) {
                sum_exp += std::exp(val - max_val); // For numerical stability
            }
            for (float& val : row) {
                val = std::exp(val - max_val) / sum_exp; // Softmax formula
            }
        }
    } else {
        throw std::invalid_argument("Unsupported activation type: " + activation_type);
    }

    return outputs;
}

std::vector<std::vector<float>> ActivationLayer::backward(const std::vector<std::vector<float>>& gradient) {
    std::vector<std::vector<float>> input_gradients = gradient;

    if (activation_type == "relu") {
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                input_gradients[i][j] *= (inputs[i][j] > 0) ? 1.0f : 0.0f; // Gradient of ReLU
            }
        }
    }
    else if (activation_type == "softmax") {
        // Do not modify gradients; already handled by CrossEntropyLoss
    }
    else {
        throw std::invalid_argument("Unsupported activation type: " + activation_type);
    }

    return input_gradients;
}


// DenseLayer save implementation
void DenseLayer::save(std::ostream& os) const {
    // Save the size of the weights matrix
    size_t input_size = weights.size();
    size_t output_size = weights[0].size();
    os.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    os.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

    // Save weights
    for (const auto& row : weights) {
        os.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }

    // Save biases
    os.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
}

// DenseLayer load implementation
void DenseLayer::load(std::istream& is) {
    // Load the size of the weights matrix
    size_t input_size = 0;
    size_t output_size = 0;
    is.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    is.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

    // Resize weights and biases accordingly
    weights.resize(input_size, std::vector<float>(output_size));
    biases.resize(output_size);

    // Load weights
    for (auto& row : weights) {
        is.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    }

    // Load biases
    is.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
}
