#include "dqerQueueLearning.h"
#include <iostream>

DQERQueueLearning::DQERQueueLearning(size_t _nrEpisodes, OpModellingType pOpModellingType, float _alpha, float _epsilon,
                         float _gamma)
    : Agent(_nrEpisodes, pOpModellingType, _gamma), alpha(_alpha), epsilon(_epsilon),
      targetMLP(mlp)
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
        if(expCounter > sizeExperience)
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
//    std::cout<<qValues.transpose()<<std::endl;
    size_t action = actionWithQ(qValues);
    auto [reward, canContinue] = maze->computeNextStateAndReward(static_cast<Actions>(action));
    Eigen::VectorXf newState = maze->getStateForAgent();
    if (not canContinue)
    {
        handleExperience();
        experiences.push_back({ action, reward, true, lastState, newState });

        // lastState and lastAction will probably be handled by newEpisode so they should not matter

        return false;
    }

    handleExperience();
    if(maze->getLastSwitchedLevel())
    {
        experiences.push_back({ action, reward, true, lastState, newState });
    }
    else
    {
        experiences.push_back({ action, reward, false, lastState, newState });
    }
    // do I need lastState somewhere? Since learning rate is 1 it reduces in the equation and
    // the backprop is already done on the deltas from lastState
    // check if d_oldstate should be updated even if we can't continue
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
size_t DQERQueueLearning::actionWithQ(Eigen::VectorXf const &qVals)
{
    bool explore = globalRng.getUniReal01() < epsilon;
    size_t choice;
    if (explore)
    {
        choice = globalRng.getUniReal01() * NR_ACTIONS;
    }
    else
    {
        qVals.maxCoeff(&choice);
    }

    return choice;
}

DQERQueueLearning::~DQERQueueLearning()
{
}