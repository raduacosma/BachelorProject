#include "qLearning.h"
#include <iostream>

QLearning::QLearning(size_t _nrEpisodes, OpModellingType pOpModellingType, float _alpha, float _epsilon, float _gamma) // TODO: check how size is passed
    : Agent(_nrEpisodes, pOpModellingType), alpha(_alpha), epsilon(_epsilon), gamma(_gamma)
{
}
void QLearning::newEpisode()
{
    Agent::newEpisode();
}
bool QLearning::performOneStep()
{
    Eigen::VectorXf qValues = mlp.feedforward(lastState);
    size_t action = actionWithQ(qValues);
    auto [reward, canContinue] = maze->computeNextStateAndReward(static_cast<Actions>(action));
    Eigen::VectorXf newState = maze->getStateForAgent();
//    float lastBestQValue = qValues(lastAction);
    if (not canContinue)
    {
        float diff = reward;
        qValues(action) = diff;
        mlp.update(qValues);
        // lastState and lastAction will probably be handled by newEpisode so they should not matter
        return false;
    }
    Eigen::VectorXf newQValues= mlp.predict(newState);
    float diff = reward + gamma*newQValues.maxCoeff();
    qValues(action) = diff;
//    std::cout<<diff<<" "<<reward<<" "<<gamma<<" "<<newQValues(newAction)<<std::endl;
    mlp.update(qValues);
    // do I need lastState somewhere? Since learning rate is 1 it reduces in the equation and
    // the backprop is already done on the deltas from lastState
    // check if d_oldstate should be updated even if we can't continue
    lastState = newState;
    return true;
}
size_t QLearning::actionWithQ(Eigen::VectorXf const &qVals)
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

QLearning::~QLearning()
{
}