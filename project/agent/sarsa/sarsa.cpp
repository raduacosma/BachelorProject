#include "sarsa.h"
#include <iostream>
#include <utility>

Sarsa::Sarsa(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
             MLPParams opponentMLP, size_t _nrEpisodes, size_t pNrEpisodesToEpsilonZero,
             OpModellingType pOpModellingType, float pAlpha, float pEpsilon,
             float pGamma) // TODO: check how size is passed
    : Agent(opTrackParams, agentMonteCarloParams, std::move(agentMLP), std::move(opponentMLP), _nrEpisodes,
            pNrEpisodesToEpsilonZero, pOpModellingType, pAlpha, pEpsilon, pGamma)
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

    Eigen::VectorXf newQValues = mlp.predict(newState);
    size_t newAction =
        actionWithQ(newQValues); // this needs to be only predict, and store the activations for next time
    float diff = reward + gamma * newQValues(newAction);
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

Sarsa::~Sarsa()
{
}