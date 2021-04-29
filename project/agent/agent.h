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

    Eigen::VectorXf oldState;
    size_t lastAction;
    size_t nrEpisodes;
    std::vector<float> rewards;
    std::vector<size_t> hasDied;

    public:
    Agent(size_t _nrEpisodes);
    virtual ~Agent() = default;

    void run();
    bool performOneStep();
    std::vector<float> & getRewards();
    std::vector<size_t> & getHasDied();
    void setMaze(SimContainer *maze);
    float getRunReward();

    protected:
    virtual Actions action(Eigen::VectorXf const &state) = 0;
    virtual void giveFeedback(float reward, Eigen::VectorXf const &newStateIdx) = 0;
    virtual void newEpisode();
};


#endif
