#include "dqerQueueLearning.h"
#include <iostream>
#include <utility>

DQERQueueLearning::DQERQueueLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams,
                                     MLPParams agentMLP, MLPParams opponentMLP, ExpReplayParams expReplayParams,
                                     size_t _nrEpisodes, size_t pNrEpisodesToEpsilonZero,
                                     OpModellingType pOpModellingType, float pEpsilon, float pGamma)
    : Agent(opTrackParams, agentMonteCarloParams, std::move(agentMLP), std::move(opponentMLP), _nrEpisodes,
            pNrEpisodesToEpsilonZero, pOpModellingType, pEpsilon, pGamma),
      targetMLP(mlp), cSwapPeriod(expReplayParams.cSwapPeriod), miniBatchSize(expReplayParams.miniBatchSize),
      sizeExperience(expReplayParams.sizeExperience)
{
}
void DQERQueueLearning::newEpisode()
{
    Agent::newEpisode();
}
bool DQERQueueLearning::performOneStep()
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
    //    Eigen::VectorXf qValues = mlp.predict(lastState);
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
void DQERQueueLearning::handleExperience()
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
void DQERQueueLearning::updateWithExperienceReplay()
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
            size_t maxIdx;
            mlp.predict(exp.newState).maxCoeff(&maxIdx);
            float expDiff = exp.reward + gamma * expNewQValues(maxIdx);
            expQValues(exp.action) = expDiff;
        }
        mlp.update(expQValues, MLPUpdateType::MINIBATCH);
    }
    mlp.updateMiniBatchWeights();
}

DQERQueueLearning::~DQERQueueLearning()
{
}