#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void update(std::vector<std::vector<float>>& weights, const std::vector<std::vector<float>>& gradients) = 0;
    virtual void update(std::vector<float>& biases, const std::vector<float>& gradients) = 0;
};

class SGDOptimizer : public Optimizer {
private:
    float learning_rate;

public:
    explicit SGDOptimizer(float lr) : learning_rate(lr) {}

    void update(std::vector<std::vector<float>>& weights, const std::vector<std::vector<float>>& gradients) override;
    void update(std::vector<float>& biases, const std::vector<float>& gradients) override;
};

#endif // OPTIMIZER_HPP
