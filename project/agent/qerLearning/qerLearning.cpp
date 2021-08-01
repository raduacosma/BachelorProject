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

#include "qerLearning.h"
#include <iostream>
#include <utility>

QERLearning::QERLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
                         MLPParams opponentMLP, ExpReplayParams expReplayParams, size_t _nrEpisodes,
                         size_t pNrEpisodesToEpsilonZero, OpModellingType pOpModellingType, float pEpsilon,
                         float pGamma)
    : Agent(opTrackParams, agentMonteCarloParams, std::move(agentMLP), std::move(opponentMLP), _nrEpisodes,
            pNrEpisodesToEpsilonZero, pOpModellingType, pEpsilon, pGamma),
      targetMLP(mlp), cSwapPeriod(expReplayParams.cSwapPeriod), miniBatchSize(expReplayParams.miniBatchSize),
      sizeExperience(expReplayParams.sizeExperience)
{
    experiences.reserve(sizeExperience);
    for (size_t idx = 0; idx != sizeExperience; ++idx)
    {
        experiences.push_back({ 0, 0, true, Eigen::VectorXf(75), Eigen::VectorXf(75) });
    }
}
void QERLearning::newEpisode()
{
    Agent::newEpisode();
}
bool QERLearning::performOneStep()
{
    if (not shouldGatherExperience)
    {
        if (expCounter == expResetPeriod)
        {
            shouldGatherExperience = true;
            expCounter = 0;
        }
    }
    else
    {
        if (expCounter == sizeExperience)
        {
            shouldGatherExperience = false;
            expCounter = 0;
        }
    }
    if (cCounter == cSwapPeriod)
    {
        targetMLP = mlp;
        cCounter = 0;
    }
    Eigen::VectorXf qValues = mlp.feedforward(lastState);
    size_t action = actionWithQ(qValues);
    auto [reward, canContinue] = maze->computeNextStateAndReward(static_cast<Actions>(action));
    Eigen::VectorXf newState = maze->getStateForAgent();
    if (not canContinue)
    {
        if (not shouldGatherExperience)
        {
            updateWithExperienceReplay();
        }
        else
            experiences[expCounter] = { action, reward, true, lastState, newState };
        ++expCounter;
        ++cCounter;
        return false;
    }
    if (not shouldGatherExperience)
    {
        updateWithExperienceReplay();
    }
    else
        experiences[expCounter] = { action, reward, false, lastState, newState };
    lastState = newState;
    ++expCounter;
    ++cCounter;
    return true;
}
void QERLearning::updateWithExperienceReplay()
{
    mlp.initMiniBatchNablas();
    for (size_t exIdx = 0; exIdx != miniBatchSize; ++exIdx)
    {
        int idx = globalRng.getExpReplayIdx();
        Experience &exp = experiences[idx];
        Eigen::VectorXf expQValues = mlp.feedforward(exp.lastState);
        if (exp.isTerminal)
        {
            expQValues(exp.action) = exp.reward;
        }
        else
        {
            Eigen::VectorXf expNewQValues = targetMLP.predict(exp.newState);
            float expDiff = exp.reward + gamma * expNewQValues.maxCoeff();
            expQValues(exp.action) = expDiff;
        }
        mlp.update(expQValues, MLPUpdateType::MINIBATCH);
    }
    mlp.updateMiniBatchWeights();
}

QERLearning::~QERLearning()
{
}