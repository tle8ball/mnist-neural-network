#include "../include/mnist_loader.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>

// Helper function to reverse the byte order of integers
int reverse_int(int n) {
    unsigned char c1, c2, c3, c4;
    c1 = n & 255;
    c2 = (n >> 8) & 255;
    c3 = (n >> 16) & 255;
    c4 = (n >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Load MNIST images
std::vector<std::vector<float>> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    int magic_number = 0;
    int num_images = 0;
    int num_rows = 0;
    int num_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&num_images, sizeof(num_images));
    num_images = reverse_int(num_images);

    file.read((char*)&num_rows, sizeof(num_rows));
    num_rows = reverse_int(num_rows);

    file.read((char*)&num_cols, sizeof(num_cols));
    num_cols = reverse_int(num_cols);

    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_rows * num_cols; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            images[i][j] = static_cast<float>(pixel);
        }
    }

    file.close();
    return images;
}

// Load MNIST labels
std::vector<int> load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + path);
    }

    int magic_number = 0;
    int num_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&num_labels, sizeof(num_labels));
    num_labels = reverse_int(num_labels);

    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    file.close();
    return labels;
}

// Normalize images to [0, 1]
void normalize_images(std::vector<std::vector<float>>& images) {
    for (auto& image : images) {
        for (auto& pixel : image) {
            pixel /= 255.0f;
        }
    }
}

// One-hot encode labels
std::vector<std::vector<float>> one_hot_encode(const std::vector<int>& labels, int num_classes) {
    std::vector<std::vector<float>> encoded(labels.size(), std::vector<float>(num_classes, 0.0f));
    for (size_t i = 0; i < labels.size(); ++i) {
        encoded[i][labels[i]] = 1.0f;
    }
    return encoded;
}
