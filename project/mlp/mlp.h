#ifndef _INCLUDED_MLP
#define _INCLUDED_MLP

#include "../Eigen/Core"
#include <cstddef>
#include <vector>
enum class ActivationFunction
{
    LINEAR,
    SIGMOID
};
class MLP
{
    std::vector<size_t> sizes;
    size_t nrLayers;
    size_t nrWeightLayers;
    size_t nrLayersBeforeActivation;
    float learningRate;
    ActivationFunction outputActivationFunction;
    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::VectorXf> biases;

    std::vector<Eigen::MatrixXf> nablaWeights;
    std::vector<Eigen::VectorXf> nablaBiases;

    std::vector<float> lossHistory;

    std::vector<Eigen::VectorXf> activations;
    std::vector<Eigen::VectorXf> zs;

  public:
    MLP(std::vector<size_t> _sizes, float _learningRate, ActivationFunction _outputActivationFunc);
    float train(Eigen::VectorXf const &input, Eigen::VectorXf const &output);
    Eigen::VectorXf predict(Eigen::VectorXf const &input);
    [[nodiscard]] std::vector<float> const &getLossHistory() const;
    static float sigmoid(float x);
    static float sigmoidPrime(float x);
    void printWeights();
    float update(Eigen::VectorXf const &output);
    Eigen::VectorXf feedforward(Eigen::VectorXf const &input);
    float updateWithGivenDiff(Eigen::VectorXf const &diff);
};
#endif
