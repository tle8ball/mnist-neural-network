#include "../include/neural_network.hpp"
#include <fstream>
#include <stdexcept>

// Destructor to clean up dynamically allocated layers
NeuralNetwork::~NeuralNetwork() {
    for (Layer* layer : layers) {
        delete layer;
    }
}

// Add a layer to the network
void NeuralNetwork::add_layer(Layer* layer) {
    layers.push_back(layer);
}

// Forward pass through all layers
std::vector<std::vector<float>> NeuralNetwork::forward(const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> output = inputs;
    for (Layer* layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

// Backward pass through all layers
void NeuralNetwork::backward(const std::vector<std::vector<float>>& output_gradient) {
    std::vector<std::vector<float>> gradient = output_gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        gradient = (*it)->backward(gradient);
    }
}

// Update weights in all layers
void NeuralNetwork::update(Optimizer& optimizer) {
    for (Layer* layer : layers) {
        layer->update(optimizer);
    }
}

// Save model to a file
void NeuralNetwork::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving: " + filepath);
    }
    for (const Layer* layer : layers) {
        layer->save(file);
    }
    file.close();
}

// Load model from a file
void NeuralNetwork::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading: " + filepath);
    }
    for (Layer* layer : layers) {
        layer->load(file);
    }
    file.close();
}
