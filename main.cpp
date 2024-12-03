#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <omp.h>
#include "./include/mnist_loader.hpp"
#include "./include/neural_network.hpp"
#include "./include/loss.hpp"
#include "./include/optimizer.hpp"
#include "./include/utils.hpp"

int main(int argc, char* argv[]) {
    // Set the number of threads to the number of available cores
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " OpenMP threads." << std::endl;

    try {
        // Check for mode argument
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " [train|evaluate]" << std::endl;
            return 1;
        }

        std::string mode = argv[1];
        bool is_train_mode = false;
        bool is_evaluate_mode = false;

        if (mode == "train") {
            is_train_mode = true;
        } else if (mode == "evaluate") {
            is_evaluate_mode = true;
        } else {
            std::cerr << "Invalid mode. Use 'train' or 'evaluate'." << std::endl;
            return 1;
        }

        // Paths to MNIST dataset files
        const std::string train_images_path = "data/train-images.idx3-ubyte";
        const std::string train_labels_path = "data/train-labels.idx1-ubyte";
        const std::string test_images_path = "data/t10k-images.idx3-ubyte";
        const std::string test_labels_path = "data/t10k-labels.idx1-ubyte";

        // Load and preprocess MNIST dataset
        std::cout << "Loading MNIST dataset..." << std::endl;
        auto train_images = load_mnist_images(train_images_path);
        auto train_labels = load_mnist_labels(train_labels_path);
        auto test_images = load_mnist_images(test_images_path);
        auto test_labels = load_mnist_labels(test_labels_path);
        std::cout << "Dataset loaded successfully!" << std::endl;

        // Verify dataset sizes
        std::cout << "Number of training images: " << train_images.size() << std::endl;
        std::cout << "Number of training labels: " << train_labels.size() << std::endl;
        std::cout << "Number of test images: " << test_images.size() << std::endl;
        std::cout << "Number of test labels: " << test_labels.size() << std::endl;

        // Normalize image pixel values to [0, 1]
        normalize_images(train_images);
        normalize_images(test_images);

        // One-hot encode labels
        auto one_hot_train_labels = one_hot_encode(train_labels, 10);
        auto one_hot_test_labels = one_hot_encode(test_labels, 10);

        // Define the neural network architecture
        std::cout << "Initializing neural network..." << std::endl;
        NeuralNetwork model;
        model.add_layer(new DenseLayer(784, 512));  // Input to Hidden Layer
        model.add_layer(new ActivationLayer("relu"));  // Activation Function
        model.add_layer(new DenseLayer(512, 512));   // Hidden Layer
        model.add_layer(new ActivationLayer("relu"));  // Activation Function
        model.add_layer(new DenseLayer(512, 512));   // Hidden Layer
        model.add_layer(new ActivationLayer("relu"));  // Activation Function
        model.add_layer(new DenseLayer(512, 512));   // Hidden Layer
        model.add_layer(new ActivationLayer("relu"));  // Activation Function
        model.add_layer(new DenseLayer(512, 10));    // Hidden to Output Layer
        model.add_layer(new ActivationLayer("softmax"));  // Softmax Activation

        if (is_evaluate_mode) {
            // Load the saved model
            std::cout << "Loading the saved model from mnist_model.bin..." << std::endl;
            model.load("mnist_model.bin");
            std::cout << "Model loaded successfully!" << std::endl;
        }

        if (is_train_mode) {
            // Loss function
            CrossEntropyLoss loss_function;

            // Optimizer
            SGDOptimizer optimizer(0.01);  // Learning rate = 0.01

            // Training parameters
            const int epochs = 10;
            const int batch_size = 32;

            // Training loop
            std::cout << "Starting training..." << std::endl;
            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs << " started." << std::endl;
                float epoch_loss = 0.0;
                int correct = 0;

                for (size_t i = 0; i < train_images.size(); i += batch_size) {
                    if (i % (batch_size * 100) == 0) { // Print every 100 batches
                        std::cout << "Processing batch " << (i / batch_size) << "/" << (train_images.size() / batch_size) << std::endl;
                    }

                    // Create mini-batch
                    size_t end = std::min(i + batch_size, train_images.size());
                    std::vector<std::vector<float>> batch_inputs(train_images.begin() + i, train_images.begin() + end);
                    std::vector<std::vector<float>> batch_labels(one_hot_train_labels.begin() + i, one_hot_train_labels.begin() + end);

                    // Forward pass
                    auto predictions = model.forward(batch_inputs);

                    // Calculate loss and accumulate
                    epoch_loss += loss_function.calculate_loss(predictions, batch_labels);

                    // Backward pass
                    auto gradients = loss_function.calculate_gradient(predictions, batch_labels);
                    model.backward(gradients);

                    // Update weights
                    model.update(optimizer);

                    // Calculate accuracy for the batch
                    correct += calculate_batch_accuracy(predictions, batch_labels);
                }

                // Log epoch metrics
                std::cout << "Epoch [" << (epoch + 1) << "/" << epochs << "] - Loss: " << epoch_loss / train_images.size()
                          << ", Accuracy: " << (static_cast<float>(correct) / train_images.size()) * 100.0 << "%" << std::endl;
            }

            // Save the model
            model.save("mnist_model.bin");
            std::cout << "Model saved to mnist_model.bin" << std::endl;
        }

        if (is_evaluate_mode) {
            // Evaluate on the test set
            std::cout << "Evaluating on test set..." << std::endl;
            CrossEntropyLoss loss_function; // Instantiate loss function for evaluation
            auto test_predictions = model.forward(test_images);
            float test_loss = loss_function.calculate_loss(test_predictions, one_hot_test_labels);
            int test_correct = calculate_batch_accuracy(test_predictions, one_hot_test_labels);

            std::cout << "Test Loss: " << test_loss / test_images.size()
                      << ", Test Accuracy: " << (static_cast<float>(test_correct) / test_images.size()) * 100.0 << "%" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
