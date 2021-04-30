#include "sarsa.h"
#include <iostream>

Sarsa::Sarsa(size_t _nrEpisodes, float _alpha, float _epsilon, float _gamma) // TODO: check how size is passed
    : Agent(_nrEpisodes), alpha(_alpha), epsilon(_epsilon), gamma(_gamma), mlp({75,192,4},0.001,ActivationFunction::LINEAR)
{
}
void Sarsa::newEpisode()
{
    Agent::newEpisode();
    lastAction = actionWithQ(mlp.predict(lastState));
}
bool Sarsa::performOneStep()
{
    lastQValues = mlp.feedforward(lastState);
    auto [reward, canContinue] = maze->computeNextStateAndReward(static_cast<Actions>(lastAction));
    Eigen::VectorXf newState = maze->getStateForAgent();
//    float lastBestQValue = lastQValues(lastAction);
    if (not canContinue)
    {
        float diff = reward;
        lastQValues(lastAction) = diff;
        mlp.update(lastQValues);
        // lastState and lastAction will probably be handled by newEpisode so they should not matter
        return false;
    }
    Eigen::VectorXf newQValues= mlp.predict(newState);
    size_t newAction = actionWithQ(newQValues);  // this needs to be only predict, and store the activations for next time
    float diff = reward + gamma*newQValues(newAction);
    lastQValues(lastAction) = diff;
//    std::cout<<diff<<" "<<reward<<" "<<gamma<<" "<<newQValues(newAction)<<std::endl;
    mlp.update(lastQValues);
    // do I need lastState somewhere? Since learning rate is 1 it reduces in the equation and
    // the backprop is already done on the deltas from lastState
    // check if d_oldstate should be updated even if we can't continue
    lastState = newState;
    lastAction = newAction;
    return true;
}
size_t Sarsa::actionWithQ(Eigen::VectorXf const &qVals)
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

Sarsa::~Sarsa()
{
}