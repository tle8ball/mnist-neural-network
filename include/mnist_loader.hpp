#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <vector>
#include <string>

// Load MNIST images
std::vector<std::vector<float>> load_mnist_images(const std::string& path);

// Load MNIST labels
std::vector<int> load_mnist_labels(const std::string& path);

// Normalize image pixel values to [0, 1]
void normalize_images(std::vector<std::vector<float>>& images);

// One-hot encode labels
std::vector<std::vector<float>> one_hot_encode(const std::vector<int>& labels, int num_classes);

#endif // MNIST_LOADER_HPP
