# C++ MNIST Neural Network

A simple C++ neural network for classifying MNIST digits, featuring custom layers, activation functions, and model saving/loading.
Test Accuracy Achieved by Neural Network: 87.95%

## Features

- **Custom Layers:** Fully connected (Dense) layers
- **Activation Functions:** ReLU and Softmax
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Model Serialization:** Save and load trained models
- **Modes:** Train, Evaluate, Inference

## Requirements

- C++11 or higher
- `g++` compiler
- MNIST dataset files placed in `data/` directory:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/tle8ball/mnist-neural-network.git
   cd mnist-neural-network
   ```

2. **Compile the Program:**
   ```bash
   g++ -Wall main.cpp src/*.cpp -o mnist_nn
   ```

## Usage

Run the program in one of the following modes:

- **Train the Model:**

  ```bash
  ./mnist_nn.exe train
  ```

- **Evaluate the Model:**

  ```bash
  ./mnist_nn.exe evaluate
  ```

- **Perform Inference:**
  ```bash
  ./mnist_nn.exe inference
  ```
