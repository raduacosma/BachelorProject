#ifndef _INCLUDED_AGENT
#define _INCLUDED_AGENT

#include <cstddef>

#include "../simContainer/simContainer.h"
#include <vector>
#include "../simState/actions.h"


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
    float d_runReward;

    size_t d_time = 0;
    SimContainer *d_maze;         // The maze the agent is navigating
    size_t d_stateSpace = 0;

    size_t d_oldstate;
    size_t d_lastAction;
    size_t d_nrEpisodes;
    std::vector<float> d_rewards;
    std::vector<size_t> d_hasDied;

    float EPSILON;

    public:
    Agent(size_t nrEpisodes, float epsilon);
    virtual ~Agent() = default;

    void run();
    bool performOneStep();
    void initialState(size_t state);
    std::vector<float> & rewards();
    std::vector<size_t> & hasDied();
    void maze(SimContainer *maze);
    float runReward();
    virtual void stateSpaceSize(size_t size) = 0;

    protected:
    virtual Actions action(size_t stateIdx) = 0;
    virtual void giveFeedback(float reward, size_t newStateIdx) = 0;
    virtual void newEpisode(size_t initialState);
};


#endif
