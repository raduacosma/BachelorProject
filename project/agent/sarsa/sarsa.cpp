#include "sarsa.h"
#include <iostream>
#include <utility>

Sarsa::Sarsa(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
             MLPParams opponentMLP, size_t _nrEpisodes, size_t pNrEpisodesToEpsilonZero,
             OpModellingType pOpModellingType, float pEpsilon,
             float pGamma)
    : Agent(opTrackParams, agentMonteCarloParams, std::move(agentMLP), std::move(opponentMLP), _nrEpisodes,
            pNrEpisodesToEpsilonZero, pOpModellingType, pEpsilon, pGamma)
{
}
void Sarsa::newEpisode()
{
    Agent::newEpisode();
    //    lastAction = actionWithQ(mlp.predict(lastState));
    lastAction = actionWithQ(MonteCarloAllActions());
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
        currentEpisodeAgentLoss += mlp.update(lastQValues);
        return false;
    }

    //    Eigen::VectorXf newQValues = mlp.predict(newState);
    Eigen::VectorXf newQValues = MonteCarloAllActions();
    size_t newAction =
        actionWithQ(newQValues); // this needs to be only predict, and store the activations for next time
    if (maze->getLastSwitchedLevel())
    {
        float diff = reward;
        lastQValues(lastAction) = diff;
    }
    else
    {
        float diff = reward + gamma * newQValues(newAction);
        lastQValues(lastAction) = diff;
    }
    currentEpisodeAgentLoss += mlp.update(lastQValues);
    lastState = newState;
    lastAction = newAction;
    return true;
}

Sarsa::~Sarsa()
{
}