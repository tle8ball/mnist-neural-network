#ifndef LOSS_HPP
#define LOSS_HPP

#include <vector>
#include <cmath>
#include <stdexcept>

class Loss {
public:
    virtual ~Loss() = default;
    virtual float calculate_loss(const std::vector<std::vector<float>>& predictions,
                                 const std::vector<std::vector<float>>& targets) = 0;
    virtual std::vector<std::vector<float>> calculate_gradient(const std::vector<std::vector<float>>& predictions,
                                                               const std::vector<std::vector<float>>& targets) = 0;
};

class CrossEntropyLoss : public Loss {
public:
    float calculate_loss(const std::vector<std::vector<float>>& predictions,
                         const std::vector<std::vector<float>>& targets) override;

    std::vector<std::vector<float>> calculate_gradient(const std::vector<std::vector<float>>& predictions,
                                                       const std::vector<std::vector<float>>& targets) override;
};

#endif // LOSS_HPP
