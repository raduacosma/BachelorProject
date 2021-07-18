#include "simContainer.h"
#include "../agent/agent.h"
#include <iostream>
#include <sstream>
#include <tuple>
using namespace std;
SimContainer::SimContainer(std::string const &filename, Agent *agentParam, Rewards rewards,
                           SimStateParams simStateParams)
    : agent(agentParam), currSimState(0), episodeCount(0), lastReward(0)
{
    string file;
    istringstream in(filename);
    while (getline(in, file, ','))
    {
        simStates.emplace_back("goodLevels/"+file, rewards, simStateParams);
    }
    agent->setMaze(this);
}

SimState &SimContainer::getCurrentLevel()
{
    return simStates[currSimState];
}
size_t SimContainer::getNrOpponents()
{
    return simStates.size();
}
bool SimContainer::nextLevel()
{
    ++currSimState;
    if (currSimState == simStates.size())
    {
        currSimState = 0;
        return false; // no more levels
    }
    return true;
}

std::tuple<float, bool> SimContainer::computeNextStateAndReward(Actions action)
{
    auto [reward, continueStatus] = simStates[currSimState].computeNextStateAndReward(action);
    lastOpponentAction = simStates[currSimState].getLastOpponentAction();
    lastOpponentState = simStates[currSimState].getStateForOpponent();
    lastSwitchedLevel = false;
    bool canContinue = true;
    switch (continueStatus)
    {

        case SimResult::CONTINUE:
            canContinue = true;
            break;
        case SimResult::REACHED_GOAL:
            if (nextLevel())
            {
                canContinue = true;
            }
            else
            {
                canContinue = false;
                ++episodeCount;
            }
            simStates[currSimState].resetForNextEpisode();
            lastSwitchedLevel = true;
            break;
        case SimResult::KILLED_BY_OPPONENT:
            canContinue = false;
            resetNextEpisode();
            break;
    }
    lastReward = reward;
    return make_tuple(reward, canContinue);
}
void SimContainer::resetNextEpisode()
{
    goToBeginning();
    simStates[currSimState].resetForNextEpisode();
    ++episodeCount;
    lastSwitchedLevel = true;
}
