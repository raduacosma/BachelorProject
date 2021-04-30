#include "sarsa.h"
#include <iostream>

Sarsa::Sarsa(size_t _nrEpisodes, float _alpha, float _epsilon, float _gamma) // TODO: check how size is passed
    : Agent(_nrEpisodes), alpha(_alpha), epsilon(_epsilon), gamma(_gamma), mlp({75,192,4},0.1,ActivationFunction::LINEAR)
{
}
void Sarsa::newEpisode()
{
    Agent::newEpisode();
    std::tie(lastAction, lastQValue) = actionWithQ(lastState);
}
bool Sarsa::performOneStep()
{
    auto [reward, canContinue] = maze->computeNextStateAndReward(static_cast<Actions>(lastAction));
    Eigen::VectorXf newState = maze->getStateForAgent();
    if (not canContinue)
    {
        float diff = reward;
        Eigen::VectorXf target = Eigen::VectorXf::Zero(4);
        target(lastAction) = diff;
        mlp.updateWithGivenDiff(target);
        // lastState and lastAction will probably be handled by newEpisode so they should not matter
        return false;
    }
    auto [newAction,newQValue] = actionWithQ(newState);
    float diff = reward + gamma*newQValue;
    Eigen::VectorXf target = Eigen::VectorXf::Zero(4);
    target(lastAction) = diff;
    std::cout<<diff<<std::endl;
    mlp.updateWithGivenDiff(target);
    // do I need lastState somewhere? Since learning rate is 1 it reduces in the equation and
    // the backprop is already done on the deltas from lastState
    // check if d_oldstate should be updated even if we can't continue
    lastState = newState;
    lastAction = newAction;
    return true;
}
std::pair<size_t, float> Sarsa::actionWithQ(Eigen::VectorXf const &state)
{
    Eigen::VectorXf mlpOutput = mlp.feedforward(state);
    bool explore = globalRng.getUniReal01() < epsilon;
    size_t choice;
    float maxVal;
    if (explore)
    {
        choice = globalRng.getUniReal01() * NR_ACTIONS;
        maxVal = mlpOutput(choice);
    }
    else
    {
        maxVal = mlpOutput.maxCoeff(&choice);
    }

    return std::make_pair(choice,maxVal);
}

Sarsa::~Sarsa()
{
}