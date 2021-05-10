#include "agent.h"
#include "../simContainer/simContainer.h"
#include "tuple"
#include <iostream>
#include <random>

using namespace std;
Agent::Agent(size_t _nrEpisodes, OpModellingType pOpModellingType, float pGamma) // TODO: check how size is passed
    : nrEpisodes(_nrEpisodes), rewards(vector<float>(_nrEpisodes)), hasDied(vector<size_t>(_nrEpisodes)),
      mlp({ 75, 192, 4 }, 0.001, ActivationFunction::LINEAR),
      opponentMlp({ 50, 100, 4 }, 0.001, ActivationFunction::SOFTMAX), opModellingType(pOpModellingType),
      gamma(pGamma)
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
        thisEpisodeLoss = std::vector<float>();
        while (true)
        {
            ++stepCount;
            bool canContinue = performOneStep();
            totalReward += maze->getLastReward();
            handleOpponentAction();
            if (not canContinue)
            {
                //                if(isNewLevel)
                //                {
                //                    nrEpisode = nrEpisodes - 1;
                //                    break;
                //                }
                break;
            }
            if (maze->getLastSwitchedLevel())
            {
                opponentNotInit = true;
                //                isNewLevel = !isNewLevel;
                //                if(not isNewLevel)
                //                {
                //                    nrEpisode = nrEpisodes - 1;
                //                    break;
                //                }
            }
        }
        std::cout << "totalReward: " << totalReward << std::endl;
        opponentPredictionLosses.push_back(currentEpisodeLoss / stepCount);
        opponentCorrectPredictionPercentage.push_back(static_cast<float>(currentEpisodeCorrectPredictions) / stepCount);
        runReward += totalReward;
        rewards[nrEpisode] = totalReward;
    }
    // any cleanup?
}
void Agent::handleOpponentAction()
{
    if (opponentNotInit)
    {
        switch (opModellingType)
        {
            case OpModellingType::NEWEVERYTIME:
                opponentMlp.randomizeWeights();
                break;
            case OpModellingType::ONEFORALL:
                break;
        }
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
    if (newOpponentAction == opponentActionIdx)
        ++currentEpisodeCorrectPredictions;
    float currentLoss = opponentMlp.update(opponentActionTarget);
    // TODO: Be careful with end of episode and reset and such
    currentEpisodeLoss += currentLoss;
    thisEpisodeLoss.push_back(currentLoss);
    lastOpponentState = newOpponentState;
}

void Agent::setMaze(SimContainer *simCont)
{
    maze = simCont;
}
//float Agent::MonteCarloRollout(size_t m, size_t N, size_t action, Eigen::VectorXf const &agentState,
//                               Eigen::VectorXf const &opState)
//{
//    float totalReward = 0;
//    for (size_t mIdx = 0; mIdx != m; ++mIdx)
//    {
//        float rolloutReward = 0;
//        size_t i = 1;
//        Eigen::VectorXf opProbs = opponentMlp.predict(opState);
//        std::discrete_distribution<> distr({opProbs[0],opProbs[1],opProbs[2],opProbs[3]});
//        size_t opAction = distr(globalRng);
//        SimContainer copyMaze(*maze);
//        auto [reward, canContinue] = copyMaze.computeNextStateAndReward(static_cast<Actions>(action),static_cast<Actions>(opAction));
//        rolloutReward+=gamma*reward; // move gamma into agent
//        while(canContinue)
//        {
//            ++i;
//            size_t agentAction = actionWithQ(copyMaze.getStateForAgent());
//            Eigen::VectorXf innerOpProbs = opponentMlp.predict(copyMaze.getStateForOpponent());
//            std::discrete_distribution<> innerDistr({innerOpProbs[0],innerOpProbs[1],innerOpProbs[2],innerOpProbs[3]});
//            size_t innerOpAction = innerDistr(globalRng);
//            auto [innerReward, innerCanContinue] = copyMaze.computeNextStateAndReward(static_cast<Actions>(agentAction),static_cast<Actions>(innerOpAction));
//            canContinue = innerCanContinue;
//            if(not canContinue)
//            {
//                rolloutReward+=std::pow(gamma,i)*innerReward;
//            }
//            else if(i==N)
//            {
//                rolloutReward+=std::pow(gamma,i)*mlp.predict(copyMaze.getStateForAgent());
//            }
//        }
//        totalReward+=rolloutReward;
//    }
//    return totalReward/N;
//}
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
size_t Agent::actionWithQ(Eigen::VectorXf const &qVals)
{
    throw std::runtime_error("In Agent's actionWithQ, should not be here");
}
