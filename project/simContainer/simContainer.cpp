#include "simContainer.h"
#include <sstream>
#include <iostream>
#include <tuple>
#include "../agent/agent.h"
using namespace std;
SimContainer::SimContainer(std::string const &filename, Agent *agentParam)
:    agent(agentParam), currSimState(0), episodeCount(0), lastReward(0)
{
    string file;
    istringstream in(filename);
    while (getline(in,file,','))
    {
        simStates.emplace_back(file);
    }
    agent->maze(this);
    sendNrStatesToAgent();
}

void SimContainer::sendNrStatesToAgent()
{
    Position simSize = simStates[currSimState].getSimSize();
    agent->stateSpaceSize(simSize.x * simSize.y);
}
SimState &SimContainer::getCurrent()
{
    return simStates[currSimState];
}
bool SimContainer::nextLevel()
{
    ++currSimState;
    if (currSimState == simStates.size())
    {
        currSimState = 0;
//        simStates[currSimState].resetForNextEpisode();
        return false; // no more levels
    }
//    simStates[currSimState].resetForNextEpisode();
    return true;
}
void SimContainer::goToBeginning()
{
    currSimState = 0;
//    simStates[currSimState].resetForNextEpisode();
}
size_t SimContainer::mazeStateHash() const
{
    return simStates[currSimState].mazeStateHash();
}
std::tuple<float, size_t, bool>
SimContainer::computeNextStateAndReward(Actions action)
{
    auto [reward,newState,continueStatus] = simStates[currSimState].computeNextStateAndReward(action);
    bool canContinue = true;
    switch (continueStatus)
    {

        case SimResult::CONTINUE:
            canContinue = true;
            break;
        case SimResult::REACHED_GOAL:
            if(nextLevel())
            {
                canContinue = true;
            }
            else
            {
                canContinue = false;
                ++episodeCount;
            }
            simStates[currSimState].resetForNextEpisode();
            cout << "reached goal"<<endl;
            agent->maze(this);
            break;
        case SimResult::KILLED_BY_OPPONENT:
            cout<<"hit opponent"<<endl;
            canContinue = false;
            goToBeginning();
            simStates[currSimState].resetForNextEpisode();
            agent->maze(this);
            ++episodeCount;
            break;
    }
    lastReward = reward;
    return make_tuple(reward,newState,canContinue);
}
size_t SimContainer::getCurrSimState() const
{
    return currSimState;
}
size_t SimContainer::getEpisodeCount() const
{
    return episodeCount;
}
float SimContainer::getLastReward() const
{
    return lastReward;
}
