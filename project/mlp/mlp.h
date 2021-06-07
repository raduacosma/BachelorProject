#ifndef _INCLUDED_MLP
#define _INCLUDED_MLP

#include "../../Eigen/Core"
#include "../utilities/utilities.h"
#include <cstddef>
#include "../createRngObj/createRngObj.h"
#include "../randObj/randobj.h"
#include <vector>

enum class MLPUpdateType
{
    NORMAL,
    MINIBATCH
};
class MLP
{
    std::vector<size_t> sizes;
    size_t nrLayers;
    size_t nrWeightLayers;
    size_t nrLayersBeforeActivation;
    size_t miniBatchSize;
    float learningRate;
    ActivationFunction outputActivationFunction;
    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::VectorXf> biases;

    std::vector<Eigen::MatrixXf> nablaWeights;
    std::vector<Eigen::VectorXf> nablaBiases;
    std::vector<Eigen::MatrixXf> nablaWeightsMiniBatch;
    std::vector<Eigen::VectorXf> nablaBiasesMiniBatch;

    std::vector<float> lossHistory;

    std::vector<Eigen::VectorXf> activations;
    std::vector<Eigen::VectorXf> zs;
    void updateMiniBatchNablas();

  public:
    MLP(std::vector<size_t> _sizes, float _learningRate, ActivationFunction _outputActivationFunc,
        size_t pMiniBatchSize);
    float train(Eigen::VectorXf const &input, Eigen::VectorXf const &output,
                MLPUpdateType updateType = MLPUpdateType::NORMAL);
    Eigen::VectorXf predict(Eigen::VectorXf const &input);
    [[nodiscard]] std::vector<float> const &getLossHistory() const;
    static float sigmoid(float x);
    static float sigmoidPrime(float x);
    void printWeights();
    float update(Eigen::VectorXf const &output, MLPUpdateType updateType = MLPUpdateType::NORMAL);
    Eigen::VectorXf feedforward(Eigen::VectorXf const &input);
    void updateWeights();
    void updateMiniBatchWeights();
    void initMiniBatchNablas();
    void randomizeWeights();
    float predictWithLoss(Eigen::VectorXf const &input, Eigen::VectorXf const &output);
};
#endif
