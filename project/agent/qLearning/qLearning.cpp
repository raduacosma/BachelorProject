#include "qLearning.h"
#include <iostream>
#include <utility>

QLearning::QLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
                     MLPParams opponentMLP, size_t _nrEpisodes,size_t pNrEpisodesToEpsilonZero, OpModellingType pOpModellingType, float pAlpha,
                     float pEpsilon,
                     float pGamma) // TODO: check how size is passed
    : Agent(opTrackParams, agentMonteCarloParams, std::move(agentMLP), std::move(opponentMLP), _nrEpisodes,pNrEpisodesToEpsilonZero,
            pOpModellingType, pAlpha,pEpsilon,pGamma)
{
}
void QLearning::newEpisode()
{
    Agent::newEpisode();
}
bool QLearning::performOneStep()
{
    Eigen::VectorXf qValues =
        mlp.feedforward(lastState); // this probably theoretically could be replaced with maze->getStateForAgent()?

    //        Eigen::VectorXf qValues = MonteCarloAllActions(); // don't do this
    //    std::cout<<qValues.transpose()<<std::endl;
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
        // lastState and lastAction will probably be handled by newEpisode so they should not matter
        return false;
    }
    Eigen::VectorXf newQValues = mlp.predict(newState);
    float diff = reward + gamma * newQValues.maxCoeff();
    qValues(action) = diff;
    //    std::cout<<diff<<" "<<reward<<" "<<gamma<<" "<<newQValues(newAction)<<std::endl;
    mlp.update(qValues);
    // do I need lastState somewhere? Since learning rate is 1 it reduces in the equation and
    // the backprop is already done on the deltas from lastState
    // check if d_oldstate should be updated even if we can't continue
    lastState = newState;
    return true;
}

QLearning::~QLearning()
{
}