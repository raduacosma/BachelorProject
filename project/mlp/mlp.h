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

#ifndef _INCLUDED_MLP
#define _INCLUDED_MLP

#include "../../Eigen/Core"
#include "../createRngObj/createRngObj.h"
#include "../randObj/randobj.h"
#include "../utilities/utilities.h"
#include <cstddef>
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
    float regParam;

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
    bool randInit;
    void updateMiniBatchNablas();

  public:
    MLP(std::vector<size_t> _sizes, float _learningRate, float pRegParam, ActivationFunction _outputActivationFunc,
        size_t pMiniBatchSize, bool pRandInit);
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
    void updateMiniBatchWeightsWithReg();
    void updateWeightsWithReg();
    float computeLoss(Eigen::VectorXf const &prediction, Eigen::VectorXf const &output);
};
#endif
