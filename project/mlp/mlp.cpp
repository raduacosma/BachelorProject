/*
     Copyright (C) 2021  Radu Alexandru Cosma

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "mlp.h"
#include <algorithm>
#include <iostream>

MLP::MLP(std::vector<size_t> _sizes, float _learningRate, float pRegParam, ActivationFunction _outputActivationFunction,
         size_t pMiniBatchSize, bool pRandInit)
    : sizes(std::move(_sizes)), nrLayers(sizes.size()), nrWeightLayers(nrLayers - 1),
      nrLayersBeforeActivation(nrLayers - 2), miniBatchSize(pMiniBatchSize), learningRate(_learningRate),
      regParam(pRegParam), outputActivationFunction(_outputActivationFunction), randInit(pRandInit)
{
    for (size_t idx = 0; idx != nrLayers; ++idx)
    {
        activations.push_back(Eigen::VectorXf::Zero(sizes[idx]));
    }
    for (size_t idx = 1; idx != nrLayers; ++idx)
    {
        zs.push_back(Eigen::VectorXf::Zero(sizes[idx]));
    }
    for (size_t idx = 1; idx != nrLayers; ++idx)
    {
        if (randInit)
            biases.push_back(Eigen::VectorXf::NullaryExpr(sizes[idx],
                                                          [&]()
                                                          {
                                                              return globalRng.getRandomInitMLP();
                                                          }));
        else
            biases.push_back(Eigen::VectorXf::Zero(sizes[idx]));
        nablaBiases.emplace_back(sizes[idx]);
        nablaBiasesMiniBatch.emplace_back(sizes[idx]);
    }
    for (size_t x = 0, y = 1; x != nrWeightLayers and y != nrLayers; ++x, ++y)
    {
        if (randInit)
            weights.push_back(Eigen::MatrixXf::NullaryExpr(sizes[y], sizes[x],
                                                           [&]()
                                                           {
                                                               return globalRng.getRandomInitMLP();
                                                           }));
        else
        {
            std::uniform_real_distribution<float> uni{ -1.0f / std::sqrt(static_cast<float>(sizes[x])),
                                                       1.0f / std::sqrt(static_cast<float>(sizes[x])) };
            weights.push_back(Eigen::MatrixXf::NullaryExpr(sizes[y], sizes[x],
                                                           [&]()
                                                           {
                                                               return uni(globalRng.getRngEngine());
                                                           }));
        }
        nablaWeights.emplace_back(sizes[y], sizes[x]);
        nablaWeightsMiniBatch.emplace_back(sizes[y], sizes[x]);
    }
}

void MLP::randomizeWeights()
{
    for (size_t idx = 1; idx != nrLayers; ++idx)
    {
        if (randInit)
            biases[idx - 1] = Eigen::VectorXf::NullaryExpr(sizes[idx],
                                                           [&]()
                                                           {
                                                               return globalRng.getRandomInitMLP();
                                                           });
        else
            biases[idx - 1] = Eigen::VectorXf::Zero(sizes[idx]);
    }
    for (size_t x = 0, y = 1; x != nrWeightLayers and y != nrLayers; ++x, ++y)
    {
        if (randInit)
            weights[x] = Eigen::MatrixXf::NullaryExpr(sizes[y], sizes[x],
                                                      [&]()
                                                      {
                                                          return globalRng.getRandomInitMLP();
                                                      });
        else
        {

            std::uniform_real_distribution<float> uni{
                -1.0f / std::sqrt(static_cast<float>(sizes[x])), 1.0f / std::sqrt(static_cast<float>(sizes[x]))
            };
            weights[x] = Eigen::MatrixXf::NullaryExpr(sizes[y], sizes[x],
                                                      [&]()
                                                      {
                                                          return uni(globalRng.getRngEngine());
                                                      });
        }
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
Eigen::VectorXf MLP::predict(Eigen::VectorXf const &input)
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
}
float MLP::predictWithLoss(Eigen::VectorXf const &input, Eigen::VectorXf const &output)
{
    Eigen::VectorXf prediction = predict(input);
    return computeLoss(prediction, output);
}
float MLP::computeLoss(Eigen::VectorXf const &prediction, Eigen::VectorXf const &output)
{
    float loss;
    if (outputActivationFunction == ActivationFunction::SOFTMAX)
    {
        loss = -(output.array() * prediction.array().log()).sum();
    }
    else
    {
        loss = (prediction - output).array().square().mean();
    }
    return loss;
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
    activations[0] = input;
    for (size_t idx = 0; idx != nrLayersBeforeActivation; ++idx)
    {
        Eigen::VectorXf &currZs = zs[idx];
        currZs = weights[idx] * activations[idx] + biases[idx];
        activations[idx + 1] = currZs.unaryExpr(&sigmoid);
    }
    Eigen::VectorXf &currZs = zs[nrLayersBeforeActivation];
    currZs =
        weights[nrLayersBeforeActivation] * activations[nrLayersBeforeActivation] + biases[nrLayersBeforeActivation];
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
        activationRef = exps / expsSum;
        return activationRef;
    }
}
float MLP::update(Eigen::VectorXf const &output, MLPUpdateType updateType)
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
        loss = -(output.array() * activations.back().array().log()).sum();
    }
    else
    {
        loss = (activations.back() - output).array().square().mean();
    }
    Eigen::VectorXf &biasesBack = nablaBiases.back();
    biasesBack = delta;
    nablaWeights.back() = biasesBack * activations[nrLayers - 2].transpose();
    for (size_t l = 2; l != nrLayers; ++l)
    {
        size_t const idx = nrWeightLayers - l;
        Eigen::VectorXf &currNablaBiases = nablaBiases[idx];
        currNablaBiases =
            (weights[idx + 1].transpose() * nablaBiases[idx + 1]).cwiseProduct(zs[idx].unaryExpr(&sigmoidPrime));
        nablaWeights[idx] = currNablaBiases * activations[idx].transpose();
    }
    if (updateType == MLPUpdateType::NORMAL)
    {
        if (regParam > 0)
            updateWeightsWithReg();
        else
            updateWeights();
    }
    else
    {
        updateMiniBatchNablas();
    }

    return loss;
}
void MLP::initMiniBatchNablas()
{
    for (size_t idx = 1; idx != nrLayers; ++idx)
    {
        nablaBiasesMiniBatch[idx - 1] = Eigen::VectorXf::Zero(sizes[idx]);
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
    if (regParam > 0)
    {
        updateMiniBatchWeightsWithReg();
        return;
    }
    float const updateCoeff = learningRate / miniBatchSize;
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
void MLP::updateMiniBatchWeightsWithReg()
{
    float const updateCoeff = learningRate / miniBatchSize;
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        weights[idx] *= 1.0f - updateCoeff * regParam;
        weights[idx] -= updateCoeff * nablaWeightsMiniBatch[idx];
        biases[idx] -= updateCoeff * nablaBiasesMiniBatch[idx];
    }
}
void MLP::updateWeightsWithReg()
{
    for (size_t idx = 0; idx != nrWeightLayers; ++idx)
    {
        weights[idx] *= 1.0f - learningRate * regParam;
        weights[idx] -= learningRate * nablaWeights[idx];
        biases[idx] -= learningRate * nablaBiases[idx];
    }
}
float MLP::train(Eigen::VectorXf const &input, Eigen::VectorXf const &output, MLPUpdateType updateType)
{
    feedforward(input);
    return update(output, updateType);
}
