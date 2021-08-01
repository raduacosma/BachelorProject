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

#include "qerQueueLearning.h"
#include <fstream>
#include <iostream>
#include <utility>

QERQueueLearning::QERQueueLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams,
                                   MLPParams agentMLP, MLPParams opponentMLP, ExpReplayParams expReplayParams,
                                   size_t _nrEpisodes, size_t pNrEpisodesToEpsilonZero,
                                   OpModellingType pOpModellingType, float pEpsilon, float pGamma)
    : Agent(opTrackParams, agentMonteCarloParams, std::move(agentMLP), std::move(opponentMLP), _nrEpisodes,
            pNrEpisodesToEpsilonZero, pOpModellingType, pEpsilon, pGamma),
      targetMLP(mlp), cSwapPeriod(expReplayParams.cSwapPeriod), miniBatchSize(expReplayParams.miniBatchSize),
      sizeExperience(expReplayParams.sizeExperience)
{
}
void QERQueueLearning::newEpisode()
{
    Agent::newEpisode();
}
bool QERQueueLearning::performOneStep()
{
    if (shouldGatherExperience)
    {
        if (expCounter > sizeExperience)
        {
            shouldGatherExperience = false;
        }
    }
    if (cCounter == cSwapPeriod)
    {
        targetMLP = mlp;
        cCounter = 0;
    }
    //            Eigen::VectorXf qValues = mlp.predict(lastState);
    Eigen::VectorXf qValues = MonteCarloAllActions();
    size_t action = actionWithQ(qValues);
    auto [reward, canContinue] = maze->computeNextStateAndReward(static_cast<Actions>(action));
    Eigen::VectorXf newState = maze->getStateForAgent();
    if (not canContinue)
    {
        handleExperience();
        experiences.push_back({ action, reward, true, lastState, newState });
        return false;
    }

    handleExperience();
    if (maze->getLastSwitchedLevel())
    {
        experiences.push_back({ action, reward, true, lastState, newState });
    }
    else
    {
        experiences.push_back({ action, reward, false, lastState, newState });
    }
    lastState = newState;

    return true;
}
void QERQueueLearning::handleExperience()
{
    if (not shouldGatherExperience)
    {
        experiences.pop_front();
        updateWithExperienceReplay();
    }
    else
        ++expCounter;
    ++cCounter;
}
void QERQueueLearning::updateWithExperienceReplay()
{
    mlp.initMiniBatchNablas();
    float currentLoss = 0;
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
        currentLoss += mlp.update(expQValues, MLPUpdateType::MINIBATCH);
    }
    currentEpisodeAgentLoss += currentLoss;
    mlp.updateMiniBatchWeights();
}

QERQueueLearning::~QERQueueLearning()
{
}
