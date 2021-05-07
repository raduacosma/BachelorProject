#include "mlp.h"
#include <algorithm>
#include <iostream>

MLP::MLP(std::vector<size_t> _sizes, float _learningRate, ActivationFunction _outputActivationFunction, size_t pMiniBatchSize)
    : sizes(std::move(_sizes)), nrLayers(sizes.size()), nrWeightLayers(nrLayers - 1),
      nrLayersBeforeActivation(nrLayers - 2), miniBatchSize(pMiniBatchSize), learningRate(_learningRate),
      outputActivationFunction(_outputActivationFunction)
{
    for(size_t idx = 0; idx!=nrLayers; ++idx)
    {
        activations.push_back(Eigen::VectorXf::Zero(sizes[idx]));
    }
    for(size_t idx = 1; idx!=nrLayers; ++idx)
    {
        zs.push_back(Eigen::VectorXf::Zero(sizes[idx]));
    }
    for (size_t idx = 1; idx != nrLayers; ++idx)
    {
        biases.push_back(Eigen::VectorXf::Random(sizes[idx]));
        nablaBiases.emplace_back(sizes[idx]);
        nablaBiasesMiniBatch.emplace_back(sizes[idx]);
    }
    for (size_t x = 0, y = 1; x != nrWeightLayers and y != nrLayers; ++x, ++y)
    {
        weights.push_back(Eigen::MatrixXf::Random(sizes[y], sizes[x]));
        nablaWeights.emplace_back(sizes[y], sizes[x]);
        nablaWeightsMiniBatch.emplace_back(sizes[y], sizes[x]);
    }
}
void MLP::printWeights()
{
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        std::cout << "biases: " << biases[idx] << std::endl;
        std::cout << "weights: " << weights[idx] << " " << std::endl;
    }
}
std::vector<float> const &MLP::getLossHistory() const
{
    return lossHistory;
}
Eigen::VectorXf MLP::predict(const Eigen::VectorXf &input)
{
    Eigen::VectorXf activation = input;
    for (size_t idx = 0; idx != nrLayersBeforeActivation; ++idx)
    {
        activation = (weights[idx] * activation + biases[idx]).unaryExpr(&sigmoid);
    }
    if (outputActivationFunction == ActivationFunction::SIGMOID)
    {
        return (weights[nrLayersBeforeActivation] * activation + biases[nrLayersBeforeActivation]).unaryExpr(&sigmoid);
    }
    else if (outputActivationFunction == ActivationFunction::LINEAR)
    {
        return weights[nrLayersBeforeActivation] * activation + biases[nrLayersBeforeActivation];
    }
    else if (outputActivationFunction == ActivationFunction::SOFTMAX)
    {
        Eigen::VectorXf const exps =
            (weights[nrLayersBeforeActivation] * activation + biases[nrLayersBeforeActivation]).array().exp();
        float expsSum = exps.sum();
        return exps / expsSum;
    }
    throw std::runtime_error("Reached end of predict without returning");
}
float MLP::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
};
float MLP::sigmoidPrime(float x)
{
    return sigmoid(x) * (1.0f - sigmoid(x));
};

Eigen::VectorXf MLP::feedforward(Eigen::VectorXf const &input)
{
    if(input.size() == 0)
        std::cout<<"yup,size is 0"<<std::endl;
    activations[0] = input;
    for (size_t idx = 0; idx != nrLayersBeforeActivation; ++idx)
    {
        Eigen::VectorXf &currZs = zs[idx];
        currZs = weights[idx] * activations[idx] + biases[idx];
        activations[idx + 1] = currZs.unaryExpr(&sigmoid);
    }
    //    std::cout<<"activations: "<<activations.size()<<std::endl;
    //    std::cout<<"zs: "<<zs.size()<<std::endl;
    Eigen::VectorXf &currZs = zs[nrLayersBeforeActivation];
    currZs = weights[nrLayersBeforeActivation] * activations[nrLayersBeforeActivation] +
             biases[nrLayersBeforeActivation];
    Eigen::VectorXf &activationRef = activations[nrLayersBeforeActivation + 1];
    if (outputActivationFunction == ActivationFunction::SIGMOID)
    {
        activationRef = currZs.unaryExpr(&sigmoid);
        return activationRef;
    }
    else if (outputActivationFunction == ActivationFunction::LINEAR)
    {
        activationRef = currZs;
        return activationRef;
    }
    else if (outputActivationFunction == ActivationFunction::SOFTMAX)
    {
        Eigen::VectorXf const exps = currZs.array().exp();
        float expsSum = exps.sum();
        activationRef = exps/expsSum;
        return activationRef;
    }
    throw std::runtime_error("Reached end of feedforward without returning");
}
float MLP::update(Eigen::VectorXf const &output,MLPUpdateType updateType)
{
    Eigen::VectorXf delta;  // maybe directly modify nablaBiases.back()?
    if (outputActivationFunction == ActivationFunction::SIGMOID)
    {
        delta = (activations.back() - output).cwiseProduct(zs.back().unaryExpr(&sigmoidPrime));
    }
    else if (outputActivationFunction == ActivationFunction::LINEAR)
    {
        delta = (activations.back() - output);
    }
    else if (outputActivationFunction == ActivationFunction::SOFTMAX)
    {
        size_t idx;
        output.maxCoeff(&idx);
        delta = activations.back();
        delta(idx) -= 1.0f;
    }
    float loss;
    if (outputActivationFunction == ActivationFunction::SOFTMAX)
    {
        loss= -(output.array()*activations.back().array().log()).sum();
//        Eigen::VectorXf logActivations = activations.back().array().log();
//        loss = (-output.array() *  logActivations.array()-
//                (1 - output.array()) * (1 - logActivations.array()))
//                   .array()
//                   .sum(); // optimize this
    }
    else
    {
        loss = (activations.back() - output).array().square().mean();
    }
    //    std::cout<<"loss: "<<loss<<std::endl;
    Eigen::VectorXf &biasesBack = nablaBiases.back();
    biasesBack = delta;
    //    std::cout<<delta.transpose()<<std::endl;
    nablaWeights.back() = biasesBack * activations[nrLayers - 2].transpose();
    for (size_t l = 2; l != nrLayers; ++l)
    {
        size_t const idx = nrWeightLayers-l;
        Eigen::VectorXf &currNablaBiases = nablaBiases[idx];
        currNablaBiases = (weights[idx + 1].transpose() * nablaBiases[idx+1])
            .cwiseProduct(zs[idx].unaryExpr(&sigmoidPrime));
        nablaWeights[idx] = currNablaBiases * activations[idx].transpose();
    }
    if(updateType == MLPUpdateType::NORMAL)
        updateWeights();
    else
        updateMiniBatchNablas();
    return loss;
}
void MLP::initMiniBatchNablas()
{
    for (size_t idx = 1; idx != nrLayers; ++idx)
    {
        nablaBiasesMiniBatch[idx-1] = Eigen::VectorXf::Zero(sizes[idx]);
    }
    for (size_t x = 0, y = 1; x != nrWeightLayers and y != nrLayers; ++x, ++y)
    {
        nablaWeightsMiniBatch[x] = Eigen::MatrixXf::Zero(sizes[y], sizes[x]);
    }
}
void MLP::updateMiniBatchNablas()
{
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        nablaWeightsMiniBatch[idx] += nablaWeights[idx];
        nablaBiasesMiniBatch[idx] += nablaBiases[idx];
    }
}
void MLP::updateMiniBatchWeights()
{
    float const updateCoeff = learningRate/miniBatchSize;
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        weights[idx] -= updateCoeff * nablaWeightsMiniBatch[idx];
        biases[idx] -= updateCoeff * nablaBiasesMiniBatch[idx];
    }
}
void MLP::updateWeights()
{
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        weights[idx] -= learningRate * nablaWeights[idx];
        biases[idx] -= learningRate * nablaBiases[idx];
    }
}
float MLP::train(Eigen::VectorXf const &input, Eigen::VectorXf const &output, MLPUpdateType updateType)
{
    feedforward(input);
    return update(output, updateType);
}
