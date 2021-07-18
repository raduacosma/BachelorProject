#ifndef _INCLUDED_SIMCONTAINER
#define _INCLUDED_SIMCONTAINER

#include <string>
#include <vector>
class Agent;
#include "../../Eigen/Core"
#include "../simState/simState.h"
#include "../utilities/utilities.h"

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
    SimContainer() = default;
    SimContainer(std::string const &filename, Agent *agentParam, Rewards rewards, SimStateParams simStateParams);
    SimState &getCurrentLevel();
    Eigen::VectorXf getStateForAgent() const;
    std::tuple<float, bool> computeNextStateAndReward(Actions action);
    bool nextLevel();
    void goToBeginning();

    Eigen::VectorXf getStateForOpponent() const;
    void resetNextEpisode();
    Eigen::VectorXf getCurrentStateForOpponent() const;
    size_t getNrOpponents();
};
inline Eigen::VectorXf SimContainer::getCurrentStateForOpponent() const
{
    return simStates[currSimState].getStateForOpponent();
}

inline Eigen::VectorXf SimContainer::getStateForOpponent() const
{
    return lastOpponentState;
}
inline size_t SimContainer::getLastOpponentAction() const
{
    return lastOpponentAction;
}

inline void SimContainer::goToBeginning()
{
    currSimState = 0;
}
inline Eigen::VectorXf SimContainer::getStateForAgent() const
{
    return simStates[currSimState].getStateForAgent();
}
inline size_t SimContainer::getCurrSimState() const
{
    return currSimState;
}
inline size_t SimContainer::getEpisodeCount() const
{
    return episodeCount;
}
inline float SimContainer::getLastReward() const
{
    return lastReward;
}
inline bool SimContainer::getLastSwitchedLevel() const
{
    return lastSwitchedLevel;
}
#endif
