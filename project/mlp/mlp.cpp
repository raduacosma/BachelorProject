#include "mlp.h"
#include <algorithm>
#include <iostream>

MLP::MLP(std::vector<size_t> _sizes, float _learningRate,
         ActivationFunction _outputActivationFunction)
    : sizes(std::move(_sizes)), nrLayers(sizes.size()), nrWeightLayers(nrLayers - 1),
      nrLayersBeforeActivation(nrLayers - 2), learningRate(_learningRate),
      outputActivationFunction(_outputActivationFunction)
{
    for (size_t idx = 1; idx != nrLayers; ++idx)
    {
        biases.push_back(Eigen::VectorXf::Random(sizes[idx]));
        nablaBiases.emplace_back(sizes[idx]);
    }
    for (size_t x = 0, y = 1; x != nrWeightLayers and y != nrLayers; ++x, ++y)
    {
        weights.push_back(Eigen::MatrixXf::Random(sizes[y], sizes[x]));
        nablaWeights.emplace_back(sizes[y], sizes[x]);
    }
}
void MLP::printWeights()
{
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        std::cout << "biases: " << biases[idx] << std::endl;
        std::cout << "weights: " << weights[idx] << " " << weights[idx] << std::endl;
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
    Eigen::VectorXf activation = input;
    activations = std::vector<Eigen::VectorXf>{};
    activations.push_back(input);
    zs = std::vector<Eigen::VectorXf>{};
    for (size_t idx = 0; idx != nrLayersBeforeActivation; ++idx)
    {
        Eigen::VectorXf z = weights[idx] * activation + biases[idx];
        zs.push_back(z);
        activation = z.unaryExpr(&sigmoid);
        activations.push_back(activation);
    }

    if (outputActivationFunction == ActivationFunction::SIGMOID)
    {
        Eigen::VectorXf z = weights[nrLayersBeforeActivation] * activation + biases[nrLayersBeforeActivation];
        zs.push_back(z);
        activation = z.unaryExpr(&sigmoid);
        activations.push_back(activation);
        return activation;
    }
    else if (outputActivationFunction == ActivationFunction::LINEAR)
    {
        Eigen::VectorXf z = weights[nrLayersBeforeActivation] * activation + biases[nrLayersBeforeActivation];
        zs.push_back(z);
        activations.push_back(z);
        return z;
    }
    throw std::runtime_error("Reached end of feedforward without returning");
}
float MLP::updateWithGivenDiff(Eigen::VectorXf const &diff)
{
    Eigen::VectorXf delta;
    if (outputActivationFunction == ActivationFunction::SIGMOID)
    {
        delta = diff.cwiseProduct(zs.back().unaryExpr(&sigmoidPrime));
    }
    else if (outputActivationFunction == ActivationFunction::LINEAR)
    {
        delta = diff;
    }
    float loss = diff.array().square().mean();
//    std::cout<<"loss: "<<loss<<std::endl;
    nablaBiases.back() = delta;
    nablaWeights.back() = delta * activations[nrLayers - 2].transpose();
    for (size_t l = 2; l != nrLayers; ++l)
    {
        delta = (weights[nrWeightLayers - l + 1].transpose() * delta)
                .cwiseProduct(zs[nrWeightLayers - l].unaryExpr(&sigmoidPrime));
        nablaBiases[nrWeightLayers - l] = delta;
        nablaWeights[nrWeightLayers - l] = delta * activations[nrLayers - l - 1].transpose();
    }
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        weights[idx] -= learningRate * nablaWeights[idx];
        biases[idx] -= learningRate * nablaBiases[idx];
    }
    return loss;
}
float MLP::update(Eigen::VectorXf const &output)
{
    Eigen::VectorXf delta;
    if (outputActivationFunction == ActivationFunction::SIGMOID)
    {
        delta = (activations.back() - output).cwiseProduct(zs.back().unaryExpr(&sigmoidPrime));
    }
    else if (outputActivationFunction == ActivationFunction::LINEAR)
    {
        delta = (activations.back() - output);
    }
    float loss = (activations.back() - output).array().square().mean();
//    std::cout<<"loss: "<<loss<<std::endl;
    nablaBiases.back() = delta;
    nablaWeights.back() = delta * activations[nrLayers - 2].transpose();
    for (size_t l = 2; l != nrLayers; ++l)
    {
        delta = (weights[nrWeightLayers - l + 1].transpose() * delta)
            .cwiseProduct(zs[nrWeightLayers - l].unaryExpr(&sigmoidPrime));
        nablaBiases[nrWeightLayers - l] = delta;
        nablaWeights[nrWeightLayers - l] = delta * activations[nrLayers - l - 1].transpose();
    }
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        weights[idx] -= learningRate * nablaWeights[idx];
        biases[idx] -= learningRate * nablaBiases[idx];
    }
    return loss;
}
float MLP::train(Eigen::VectorXf const &input, Eigen::VectorXf const &output)
{
    Eigen::VectorXf activation = input;
    activations = std::vector<Eigen::VectorXf>{};
    activations.push_back(input);
    zs = std::vector<Eigen::VectorXf>{};

    for (size_t idx = 0; idx != nrLayersBeforeActivation; ++idx)
    {
        Eigen::VectorXf z = weights[idx] * activation + biases[idx];
        zs.push_back(z);
        activation = z.unaryExpr(&sigmoid);
        activations.push_back(activation);
    }
    Eigen::VectorXf delta;
    if (outputActivationFunction == ActivationFunction::SIGMOID)
    {
        Eigen::VectorXf z = weights[nrLayersBeforeActivation] * activation + biases[nrLayersBeforeActivation];
        zs.push_back(z);
        activation = z.unaryExpr(&sigmoid);
        activations.push_back(activation);
        delta = (activations.back() - output).cwiseProduct(zs.back().unaryExpr(&sigmoidPrime));
    }
    else if (outputActivationFunction == ActivationFunction::LINEAR)
    {
        Eigen::VectorXf z = weights[nrLayersBeforeActivation] * activation + biases[nrLayersBeforeActivation];
        zs.push_back(z);
        activations.push_back(z);
        delta = (activations.back() - output);
    }

    float loss = (activations.back() - output).array().square().mean();
    nablaBiases.back() = delta;
    nablaWeights.back() = delta * activations[nrLayers - 2].transpose();
    for (size_t l = 2; l != nrLayers; ++l)
    {
        delta = (weights[nrWeightLayers - l + 1].transpose() * delta)
                    .cwiseProduct(zs[nrWeightLayers - l].unaryExpr(&sigmoidPrime));
        nablaBiases[nrWeightLayers - l] = delta;
        nablaWeights[nrWeightLayers - l] = delta * activations[nrLayers - l - 1].transpose();
    }
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        weights[idx] -= learningRate * nablaWeights[idx];
        biases[idx] -= learningRate * nablaBiases[idx];
    }
    return loss;
}
