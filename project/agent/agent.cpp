#include "agent.h"
#include "../simContainer/simContainer.h"
#include "tuple"
#include <iostream>
using namespace std;
Agent::Agent(size_t _nrEpisodes) // TODO: check how size is passed
    : nrEpisodes(_nrEpisodes), rewards(vector<float>(_nrEpisodes)), hasDied(vector<size_t>(_nrEpisodes)),
      mlp({ 75, 192, 4 }, 0.001, ActivationFunction::LINEAR),
      opponentMlp({50,100,4},0.001,ActivationFunction::SOFTMAX)
{
}
bool Agent::performOneStep()
{
    throw std::runtime_error("In Agent's performOneStep, should not be here");
}
void Agent::run()
{
    // initialize Q values? They are set to 0 in stateSpaceSize and Marco's
    // slides say initialize "arbitrarily" while the book says terminal states
    // should be 0, so I guess initializing them all to 0 could be fine?

    // tracking stuff? avg rewards etc etc etc
    runReward = 0;

    for (size_t nrEpisode = 0; nrEpisode != nrEpisodes; ++nrEpisode)
    {
        // d_oldstate was modified from Maze so it's fine, anything else?
        newEpisode();
        currentEpisodeLoss = 0;
        currentEpisodeCorrectPredictions = 0;
        size_t stepCount = 0;
        float totalReward = 0;
        opponentNotInit = true;
        while (true)
        {
            ++stepCount;
            bool canContinue = performOneStep();
            totalReward += maze->getLastReward();
            handleOpponentAction();
            if (not canContinue)
                break;
            if(maze->getLastSwitchedLevel())
                opponentNotInit = true;
        }
        std::cout<<"totalReward: "<<totalReward<<std::endl;
        opponentPredictionLosses.push_back(currentEpisodeLoss/stepCount);
        opponentCorrectPredictionPercentage.push_back(static_cast<float>(currentEpisodeCorrectPredictions)/stepCount);
        runReward += totalReward;
        rewards[nrEpisode] = totalReward;
    }
    // any cleanup?
}
void Agent::handleOpponentAction()
{
    if(opponentNotInit)
    {
        lastOpponentState = maze->getStateForOpponent();
        opponentNotInit = false;
        return;
    }
    Eigen::VectorXf newOpponentState = maze->getStateForOpponent();
    size_t newOpponentAction = maze->getLastOpponentAction();
    Eigen::VectorXf opponentActionTarget = Eigen::VectorXf::Zero(4);
    opponentActionTarget(static_cast<size_t>(newOpponentAction)) = 1.0f;
    size_t opponentActionIdx;
    opponentMlp.feedforward(lastOpponentState).maxCoeff(&opponentActionIdx);
    //    std::cout<<opponentActionIdx<<" "<<newOpponentAction<<std::endl;
    if(newOpponentAction == opponentActionIdx)
        ++currentEpisodeCorrectPredictions;
    float currentLoss = opponentMlp.update(opponentActionTarget);
    // TODO: Be careful with end of episode and reset and such
    currentEpisodeLoss += currentLoss;
    //    thisEpisodeLoss.push_back(currentLoss);
    lastOpponentState = newOpponentState;
}

void Agent::setMaze(SimContainer *simCont)
{
    maze = simCont;
}
void Agent::newEpisode()
{
    lastState = maze->getStateForAgent();
    lastOpponentState = maze->getStateForOpponent();
    // Some algorithms require this. Empty for the others.
}

vector<float> &Agent::getRewards()
{
    return rewards;
}

vector<size_t> &Agent::getHasDied()
{
    return hasDied;
}

float Agent::getRunReward()
{
    return runReward;
}

Agent::~Agent()
{
}
vector<float> const &Agent::getOpponentPredictionLosses() const
{
    return opponentPredictionLosses;
}
vector<float> const &Agent::getOpponentCorrectPredictionPercentage() const
{
    return opponentCorrectPredictionPercentage;
}
vector<float> const &Agent::getThisEpisodeLoss() const
{
    return thisEpisodeLoss;
}
