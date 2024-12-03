cmake_minimum_required(VERSION 3.10)
project(MNIST_Neural_Network)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories for header files
include_directories(include)

# Add source files
set(SOURCES
    src/main.cpp
    src/mnist_loader.cpp
    src/neural_network.cpp
    src/layers.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/utils.cpp
)

# Create the executable
add_executable(mnist_nn ${SOURCES})

# Link libraries (if necessary)
# For now, we assume no external libraries are needed.
# You can uncomment and add libraries below if needed:
# target_link_libraries(mnist_nn PUBLIC <LIBRARY_NAME>)

# Enable warnings
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(mnist_nn PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Output message for build configuration
message(STATUS "CMake project configured successfully.")
