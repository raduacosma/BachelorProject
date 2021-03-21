#ifndef _INCLUDED_SIMCONTAINER
#define _INCLUDED_SIMCONTAINER


#include <vector>
#include <string>
class Agent;
#include "../simState/simState.h"


class SimContainer
{
    std::vector<SimState> simStates;
    Agent *agent;
    bool correctState = false;
    size_t currSimState;

    public:
    bool isCorrectState() const;

    public:
    SimContainer() = default;
    SimContainer(std::string const &filename, Agent *agentParam);
    SimState &getCurrent();
    size_t mazeStateHash() const;
    std::tuple<double, size_t, SimResult> computeNextStateAndReward(Actions action);
    void sendNrStatesToAgent();
    bool nextLevel();
    void goToBeginning();
};
#endif
