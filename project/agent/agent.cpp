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

#include "agent.h"
#include "../simContainer/simContainer.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>

using namespace std;
Agent::Agent(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
             MLPParams opponentMLP, size_t _nrEpisodes, size_t pNrEpisodesToEpsilonZero,
             OpModellingType pOpModellingType, float pEpsilon, float pGamma)
    : opTrack(opTrackParams.pValueThreshold, opTrackParams.minHistorySize, opTrackParams.maxHistorySize),
      nrEpisodes(_nrEpisodes), nrEpisodesToEpsilonZero(pNrEpisodesToEpsilonZero), rewards(vector<float>(_nrEpisodes)),
      mlp(agentMLP.sizes, agentMLP.learningRate, agentMLP.regParam, agentMLP.outputActivationFunc,
          agentMLP.miniBatchSize, agentMLP.randInit),
      opList{ MLP(opponentMLP.sizes, opponentMLP.learningRate, opponentMLP.regParam, opponentMLP.outputActivationFunc,
                  opponentMLP.miniBatchSize, agentMLP.randInit) },
      opLosses{ 0.0f }, currOp(0), opModellingType(pOpModellingType), epsilon(pEpsilon), gamma(pGamma),
      maxNrSteps(agentMonteCarloParams.maxNrSteps), nrRollouts(agentMonteCarloParams.nrRollouts),
      opMLPParams(opponentMLP), opDeathsPerEp(nrEpisodes, 0)
{
    gammaVals.push_back(1); // in order to save some i-1s in monte carlo
    float tmpGamma = gamma;
    for (size_t idx = 0; idx <= maxNrSteps; ++idx) // although gammaVals[idx+1] is added
    {
        gammaVals.push_back(tmpGamma);
        tmpGamma *= gamma;
    }
    rewards.reserve(nrEpisodes);
    opponentPredictionLosses.reserve(nrEpisodes);
    opponentCorrectPredictionPercentage.reserve(nrEpisodes);
    thisEpisodeLoss.reserve(nrEpisodes);
    learningLosses.reserve(nrEpisodes);
}
bool Agent::performOneStep()
{
    //    throw std::runtime_error("In Agent's performOneStep, should not be here");
}
void Agent::run()
{
    runReward = 0;
    float initialEpsilon = epsilon;
    float lastEpsilon = 0;
    float killedByOpponentReward = maze->getCurrentLevel().killedByOpponentReward();
    for (size_t nrEpisode = 0; nrEpisode != nrEpisodes; ++nrEpisode)
    {
        newEpisode();
        currentEpisodeOpLoss = 0;
        currentEpisodeAgentLoss = 0;
        currentEpisodeCorrectPredictions = 0;
        countFoundPredictionsCurrentEpisode = 0;
        foundCurrentEpisodeCorrectPredictions = 0;
        size_t stepCount = 0;
        float totalReward = 0;
        initOpponentMethod();
        while (true)
        {
            ++stepCount;
            bool canContinue = performOneStep();
            float receivedReward = maze->getLastReward();
            totalReward += receivedReward;
            handleOpponentAction();
            if (not canContinue)
            {
                if (receivedReward == killedByOpponentReward)
                    opDeathsPerEp[nrEpisode] = 1;
                break;
            }
            if (stepCount >= 1000) // max nr of timesteps
            {
                maze->resetNextEpisode();
                break;
            }
            if (maze->getLastSwitchedLevel())
            {
                initOpponentMethod();
            }
        }
        learningLosses.push_back(currentEpisodeAgentLoss / stepCount);
        opponentPredictionLosses.push_back(currentEpisodeOpLoss / stepCount);
        opponentCorrectPredictionPercentage.push_back(static_cast<float>(currentEpisodeCorrectPredictions) / stepCount);
        opponentFoundCorrectPredictionPercentage.push_back(static_cast<float>(foundCurrentEpisodeCorrectPredictions) /
                                                           countFoundPredictionsCurrentEpisode);
        runReward += totalReward;
        rewards[nrEpisode] = totalReward;
        if (epsilon > lastEpsilon)
            epsilon -= (initialEpsilon - lastEpsilon) / nrEpisodesToEpsilonZero;
    }
}
void Agent::handleOpponentAction()
{
    switch (opModellingType)
    {
        case OpModellingType::NEWEVERYTIME:
        case OpModellingType::ONEFORALL:
            opPredict(&OpTrack::normalOpTracking);
            break;
        case OpModellingType::KOLSMIR:
            opPredictInterLoss(&OpTrack::kolsmirOpTracking);
            break;
        case OpModellingType::BADLOSSPETTITT:
            opPredictInterLoss(&OpTrack::pettittOpTracking);
            break;
        case OpModellingType::NOTRAINPETTITT:
            opPredictInterLoss(&OpTrack::noTrainPettittOpTracking);
            break;
    }
}
void Agent::initOpponentMethod()
{
    lastOpponentState = maze->getCurrentStateForOpponent();
    for (auto &item : opLosses)
        item = 0.0f;
    switch (opModellingType)
    {
        case OpModellingType::NEWEVERYTIME:
            opList[currOp].randomizeWeights();
            break;
        case OpModellingType::ONEFORALL:
            break;
        case OpModellingType::KOLSMIR:
            opTrack.kolsmirOpInit(*this);
            break;
        case OpModellingType::BADLOSSPETTITT:
            opTrack.pettittOpInit(*this);
            break;
        case OpModellingType::NOTRAINPETTITT:
            opTrack.noTrainPettittOpInit(*this);
            break;
    }
}
void Agent::opPredict(void (OpTrack::*tracking)(Agent &agent, Eigen::VectorXf const &, Eigen::VectorXf const &, float))
{
    Eigen::VectorXf newOpponentState = maze->getStateForOpponent();
    size_t newOpponentAction = maze->getLastOpponentAction();
    Eigen::VectorXf opponentActionTarget = Eigen::VectorXf::Zero(4);
    opponentActionTarget(static_cast<size_t>(newOpponentAction)) = 1.0f;
    size_t opponentActionIdx;
    opList[currOp].feedforward(lastOpponentState).maxCoeff(&opponentActionIdx);
    if (newOpponentAction == opponentActionIdx)
        ++currentEpisodeCorrectPredictions;
    float currentLoss = opList[currOp].update(opponentActionTarget);
    currentEpisodeOpLoss += currentLoss;
    (opTrack.*tracking)(*this, lastOpponentState, opponentActionTarget, currentLoss);
    lastOpponentState = newOpponentState;
}

void Agent::opPredictInterLoss(void (OpTrack::*tracking)(Agent &agent, Eigen::VectorXf const &, Eigen::VectorXf const &,
                                                         float))
{
    Eigen::VectorXf newOpponentState = maze->getStateForOpponent();
    size_t newOpponentAction = maze->getLastOpponentAction();
    Eigen::VectorXf opponentActionTarget = Eigen::VectorXf::Zero(4);
    opponentActionTarget(static_cast<size_t>(newOpponentAction)) = 1.0f;
    Eigen::VectorXf currPrediction = opList[currOp].predict(lastOpponentState);
    size_t opponentActionIdx;
    currPrediction.maxCoeff(&opponentActionIdx);
    if (newOpponentAction == opponentActionIdx)
    {
        ++currentEpisodeCorrectPredictions;
        if (opTrack.isFoundOpModel())
            ++foundCurrentEpisodeCorrectPredictions;
    }
    float currentLoss = opList[currOp].computeLoss(currPrediction, opponentActionTarget);
    currentEpisodeOpLoss += currentLoss;
    size_t currIdx = currOp;
    if (not opTrack.isFoundOpModel())
    {
        for (size_t idx = 0, sz = opList.size() - 1; idx != sz; ++idx)
            opLosses[idx] += opList[idx].predictWithLoss(lastOpponentState, opponentActionTarget);
        int minIdx = -1;
        float minLoss = std::numeric_limits<float>::max();
        for (int idx = 0, sz = opLosses.size() - 1; idx != sz; ++idx)
            if (minLoss > opLosses[idx])
            {
                minIdx = idx;
                minLoss = opLosses[idx];
            }
        if (minIdx >= 0)
            currIdx = minIdx;
        currOp = opList.size() - 1;
    }
    else
    {
        ++countFoundPredictionsCurrentEpisode;
    }
    opList[currOp].train(lastOpponentState, opponentActionTarget);
    // currentLoss here is bad since it should be the train loss on currOp, but it was only used for old pettitt which
    // is not used anymore
    (opTrack.*tracking)(*this, lastOpponentState, opponentActionTarget, currentLoss);
    if (not opTrack.isFoundOpModel())
        currOp = currIdx;
    lastOpponentState = newOpponentState;
}
float Agent::MonteCarloRollout(size_t action)
{
    auto &rngEngine = globalRng.getRngEngine();
    float totalReward = 0;
    for (size_t mIdx = 0; mIdx != nrRollouts; ++mIdx)
    {
        float rolloutReward = 0;
        size_t i = 0;
        MonteCarloSim copyMaze(maze->getCurrentLevel());
        Eigen::VectorXf opProbs = opList[currOp].predict(copyMaze.getStateForOpponent());
        size_t opAction;
        if (nrRollouts == 1)
        {
            opProbs.maxCoeff(&opAction);
        }
        else
        {
            std::discrete_distribution<> distr({ opProbs[0], opProbs[1], opProbs[2], opProbs[3] });
            opAction = distr(rngEngine);
        }
        auto [reward, canContinue] =
            copyMaze.computeNextStateAndReward(static_cast<Actions>(action), static_cast<Actions>(opAction));
        rolloutReward += reward;
        if (canContinue != SimResult::KILLED_BY_OPPONENT and canContinue != SimResult::REACHED_GOAL and maxNrSteps > 0)
        {
            while (true)
            {
                ++i;

                Eigen::VectorXf innerOpProbs = opList[currOp].predict(copyMaze.getStateForOpponent());
                size_t innerOpAction;
                if (nrRollouts == 1)
                {
                    innerOpProbs.maxCoeff(&innerOpAction);
                }
                else
                {
                    std::discrete_distribution<> innerDistr(
                        { innerOpProbs[0], innerOpProbs[1], innerOpProbs[2], innerOpProbs[3] });
                    innerOpAction = innerDistr(rngEngine);
                }
                size_t agentAction;
                float lastStateVal = mlp.predict(copyMaze.getStateForAgent()).maxCoeff(&agentAction);
                auto [innerReward, innerCanContinue] = copyMaze.computeNextStateAndReward(
                    static_cast<Actions>(agentAction), static_cast<Actions>(innerOpAction));
                rolloutReward += gammaVals[i] * innerReward;
                if (innerCanContinue == SimResult::KILLED_BY_OPPONENT or innerCanContinue == SimResult::REACHED_GOAL)
                {
                    break;
                }
                if (i == maxNrSteps)
                {
                    rolloutReward += gammaVals[i] * lastStateVal;
                    break;
                }
            }
        }
        else if (canContinue == SimResult::CONTINUE and maxNrSteps == 0)
        {
            rolloutReward += mlp.predict(copyMaze.getStateForAgent()).maxCoeff();
        }
        totalReward += rolloutReward;
    }
    return totalReward / nrRollouts;
}

size_t Agent::actionWithQ(Eigen::VectorXf const &qVals) const
{
    bool explore = globalRng.getUniReal01() < epsilon;
    size_t choice;
    if (explore)
    {
        choice = globalRng.getUniReal01() * NR_ACTIONS;
    }
    else
    {
        float maxVal = qVals.maxCoeff();
        std::vector<size_t> maxIdxs;
        maxIdxs.reserve(4);
        for (size_t idx = 0; idx != NR_ACTIONS; ++idx)
            if (qVals[idx] == maxVal)
                maxIdxs.push_back(idx);
        choice = maxIdxs[globalRng.getUniReal01() * maxIdxs.size()];
    }

    return choice;
}
Eigen::VectorXf Agent::MonteCarloAllActions()
{
    Eigen::VectorXf estimatedQValues(4);
    for (size_t idx = 0; idx != NR_ACTIONS; ++idx)
        estimatedQValues[idx] = MonteCarloRollout(idx);
    return estimatedQValues;
}
void Agent::newEpisode()
{
    lastState = maze->getStateForAgent();
    lastOpponentState = maze->getStateForOpponent();
}

Agent::~Agent()
{
}

float Agent::getOpDeathPercentage() const
{
    size_t count = 0;
    for (auto const &item : opDeathsPerEp)
        if (item == 1)
            ++count;
    return static_cast<float>(count) / nrEpisodes;
}
vector<size_t> const &Agent::getOpDeathsPerEp() const
{
    return opDeathsPerEp;
}

size_t Agent::getPredictedNrOfOpponents() const
{
    return opList.size();
}
vector<float> const &Agent::getOpponentFoundCorrectPredictionPercentage() const
{
    return opponentFoundCorrectPredictionPercentage;
}
vector<size_t> const &Agent::getPredictedOpponentType() const
{
    return predictedOpponentType;
}
vector<size_t> const &Agent::getActualOpponentType() const
{
    return actualOpponentType;
}
