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
        //        std::shuffle(simStates.begin(),simStates.end(),globalRng.getRngEngine());
        currSimState = 0;
        //        simStates[currSimState].resetForNextEpisode();
        return false; // no more levels
    }
    //    simStates[currSimState].resetForNextEpisode();
    return true;
}

std::tuple<float, bool> SimContainer::computeNextStateAndReward(Actions action)
{
    auto [reward, continueStatus] = simStates[currSimState].computeNextStateAndReward(action);
    lastOpponentAction = simStates[currSimState].getLastOpponentAction();
    // should these really be before the level is changed?
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
//            cout << "reached goal" << endl;
            lastSwitchedLevel = true;
            //            agent->maze(this);
            break;
        case SimResult::KILLED_BY_OPPONENT:
//            cout << "hit opponent" << endl;
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
    //            agent->maze(this); // needed? since everything goes through simContainer probably not
    ++episodeCount;
    lastSwitchedLevel = true;
}
