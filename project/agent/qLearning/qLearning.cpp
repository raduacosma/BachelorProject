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

#include "qLearning.h"
#include <iostream>
#include <utility>

QLearning::QLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
                     MLPParams opponentMLP, size_t _nrEpisodes, size_t pNrEpisodesToEpsilonZero,
                     OpModellingType pOpModellingType, float pEpsilon, float pGamma)
    : Agent(opTrackParams, agentMonteCarloParams, std::move(agentMLP), std::move(opponentMLP), _nrEpisodes,
            pNrEpisodesToEpsilonZero, pOpModellingType, pEpsilon, pGamma)
{
}
void QLearning::newEpisode()
{
    Agent::newEpisode();
}
bool QLearning::performOneStep()
{
    Eigen::VectorXf qValues = mlp.feedforward(lastState);

    //        Eigen::VectorXf qValues = MonteCarloAllActions();
    size_t action = actionWithQ(qValues);
    //    size_t action = actionWithQ(MonteCarloAllActions());
    auto [reward, canContinue] = maze->computeNextStateAndReward(static_cast<Actions>(action));
    Eigen::VectorXf newState = maze->getStateForAgent();
    //    float lastBestQValue = qValues(lastAction);
    if (not canContinue)
    {
        float diff = reward;
        qValues(action) = diff;
        mlp.update(qValues);
        return false;
    }
    Eigen::VectorXf newQValues = mlp.predict(newState);
    float diff = reward + gamma * newQValues.maxCoeff();
    qValues(action) = diff;
    mlp.update(qValues);

    lastState = newState;
    return true;
}

QLearning::~QLearning()
{
}