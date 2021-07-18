#ifndef _INCLUDED_MONTECARLOSIM
#define _INCLUDED_MONTECARLOSIM

#include "../../Eigen/Core"
#include "../utilities/utilities.h"
#include <deque>
#include <string>
#include <vector>

class Agent;

#include "../agent/agent.h"
#include "../simState/simState.h"

class MonteCarloSim
{

    Position agentPos;
    Position simSize;
    Position goalPos;
    size_t traceSize;
    size_t agentVisionGridSize;
    size_t agentVisionGridSideSize; // visionGridSize*2+1
    size_t agentStateSize;          // visionGridSizeSize^2
    size_t opponentVisionGridSize;
    size_t opponentVisionGridSideSize; // visionGridSize*2+1
    size_t opponentStateSize;          // visionGridSizeSize^2

    std::vector<Position> const &walls;

    // REWARDS
    float d_outOfBoundsReward;
    float d_reachedGoalReward;
    float d_killedByOpponentReward;
    float d_normalReward;
    std::deque<Position> opponentTrace;

  public:
    MonteCarloSim(SimState const &simState);
    std::tuple<float, SimResult> computeNextStateAndReward(Actions action, Actions opAction);
    Eigen::VectorXf getStateForAgent() const;
    Eigen::VectorXf getStateForOpponent() const;
    void updateOpPos(Actions opAction);

  private:
    std::pair<float, SimResult> updateAgentPos(Actions action);
    Position computeNewPos(Actions currAction, Position pos);
};

#endif
