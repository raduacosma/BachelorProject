#ifndef _INCLUDED_AGENT
#define _INCLUDED_AGENT

#include <cstddef>

#include "../simContainer/simContainer.h"
#include <vector>
#include "../simState/actions.h"
#include "../Eigen/Core"


enum class AgentType
{
    QLEARNING,
    DQLEARNING,
    SARSA,
    EXPECTEDSARSA
};

class Agent
{
    protected:

    static constexpr size_t NR_ACTIONS = 4; // Hardcoded number of actions
    size_t d_killedByAshTime = 500;
    float Q_0 = 0;
    float runReward;

    size_t simTime = 0;
    SimContainer *maze;         // The maze the agent is navigating

    Eigen::VectorXf lastState;
    size_t lastAction;
    size_t nrEpisodes;
    std::vector<float> rewards;
    std::vector<size_t> hasDied;

    public:
    explicit Agent(size_t _nrEpisodes);
    virtual ~Agent();

    void run();

    std::vector<float> & getRewards();
    std::vector<size_t> & getHasDied();
    void setMaze(SimContainer *maze);
    float getRunReward();

    virtual bool performOneStep();

  protected:
    virtual void newEpisode();
};


#endif
