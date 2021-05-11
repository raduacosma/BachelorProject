#ifndef _INCLUDED_SIMCONTAINER
#define _INCLUDED_SIMCONTAINER


#include <vector>
#include <string>
class Agent;
#include "../simState/simState.h"
#include "../../Eigen/Core"


class SimContainer
{
    std::vector<SimState> simStates;
    Agent *agent;
    size_t currSimState;
    size_t episodeCount;
    float lastReward;
    size_t lastOpponentAction;
    Eigen::VectorXf lastOpponentState;
    bool lastSwitchedLevel;

  public:
    bool getLastSwitchedLevel() const;

  public:
    float getLastReward() const;
    size_t getLastOpponentAction() const;

    public:
    size_t getCurrSimState() const;
    size_t getEpisodeCount() const;


    public:
    SimContainer() = default;     // needed? probably not since I removed those move stuff and rely on unique_pt
    SimContainer(std::string const &filename, Agent *agentParam);
    SimState &getCurrentLevel();
    Eigen::VectorXf getStateForAgent() const;
    std::tuple<float, bool> computeNextStateAndReward(Actions action);
    bool nextLevel();
    void goToBeginning();

    Eigen::VectorXf getStateForOpponent() const;
};


#endif
