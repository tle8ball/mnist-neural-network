#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include "layers.hpp"
#include "optimizer.hpp"

class NeuralNetwork {
private:
    std::vector<Layer*> layers;  // Vector of pointers to layers

public:
    // Constructor and Destructor
    NeuralNetwork() = default;
    ~NeuralNetwork();

    // Add a layer to the network
    void add_layer(Layer* layer);

    // Forward pass
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs);

    // Backward pass
    void backward(const std::vector<std::vector<float>>& output_gradient);

    // Update weights
    void update(Optimizer& optimizer);

    // Save and load model
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
};

#endif // NEURAL_NETWORK_HPP
