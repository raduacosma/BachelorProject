#ifndef _INCLUDED_MONTECARLOSIM
#define _INCLUDED_MONTECARLOSIM

#include "../../Eigen/Core"
#include "../utilities/utilities.h"
#include "imgui.h"
#include <deque>
#include <string>
#include <vector>

class Agent;

#include "../simState/simState.h"
#include "../agent/agent.h"

class MonteCarloSim
{


    Position agentPos;
    Position simSize;
    Position goalPos;
    size_t traceSize;
    size_t visionGridSize;  // TODO: change these in the constructor
    size_t visionGridSideSize;  // visionGridSize*2+1
    size_t agentStateSize;     // visionGridSizeSize^2

    std::vector<Position> const &walls;



    // REWARDS TODO: decide on these
    float d_outOfBoundsReward;
    float d_reachedGoalReward;
    float d_killedByOpponentReward;
    float d_normalReward;
    std::deque<Position> opponentTrace;

  public:
    MonteCarloSim(SimState const &simState);
    // this also moves the agent
    std::tuple<float, SimResult> computeNextStateAndReward(Actions action);
    Eigen::VectorXf getStateForAgent() const;
    Eigen::VectorXf getStateForOpponent() const;
    void updateOpPos(Actions opAction);
  private:
    std::pair<float, SimResult> updateAgentPos(Actions action);
    Position computeNewPos(Actions currAction, Position pos);

};




#endif
