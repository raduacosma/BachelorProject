#include "simContainer.h"
#include <sstream>
#include <iostream>
#include <tuple>
#include "../agent/agent.h"
using namespace std;
SimContainer::SimContainer(std::string const &filename, Agent *agentParam)
:    agent(agentParam)
{
    string file;
    istringstream in(filename);
    while (getline(in,file,','))
    {
        cout<<file<<endl;
        simStates.emplace_back(file);
    }
    currSimState = 0;
    agent->maze(this);
    sendNrStatesToAgent();
    correctState = true;
}
void SimContainer::sendNrStatesToAgent()
{
    Position simSize = simStates[currSimState].getSimSize();
    agent->stateSpaceSize(simSize.x * simSize.y);
}
bool SimContainer::isCorrectState() const
{
    return correctState;
}
SimState &SimContainer::getCurrent()
{
    return simStates[currSimState];
}
//size_t SimContainer::mazeStateHash() const
//{
//    return (&simStates[currSimState])->mazeStateHash();
//}
//std::tuple<double, size_t, bool>
//SimContainer::computeNextStateAndReward(Actions action)
//{
//    return (&simStates[currSimState])->computeNextStateAndReward(action);
//}
