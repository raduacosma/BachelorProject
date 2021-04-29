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
    float learningRate;
    size_t nrEpisodes;
    ActivationFunction outputActivationFunction;
    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::VectorXf> biases;

    std::vector<Eigen::MatrixXf> nablaWeights;
    std::vector<Eigen::VectorXf> nablaBiases;

    std::vector<float> lossHistory;

  public:
    MLP(std::vector<size_t> _sizes, float _learningRate, size_t _nrEpisodes, ActivationFunction _outputActivationFunc);
    float train(Eigen::VectorXf const &input, Eigen::VectorXf const &output);
    float infer(Eigen::VectorXf const &input); // or maybe vector of all options/ action nr?
    [[nodiscard]] std::vector<float> const &getLossHistory() const;
    static float sigmoid(float x);
    static float sigmoidPrime(float x);
    void printWeights();
};
#endif
