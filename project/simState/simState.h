#ifndef _INCLUDED_SIMSTATE
#define _INCLUDED_SIMSTATE

#include "../../Eigen/Core"
#include "../utilities/utilities.h"
#include "imgui.h"
#include <string>
#include <vector>

class Agent;

#include "actions.h"

class SimState
{
    friend class MonteCarloSim;

  private:
    std::vector<std::vector<ImVec4>> stateRepresentation;
    ImVec2 canvasStepSize;
    ImVec2 canvasBegPos;
    ImVec2 canvasEndPos;

    Position simSize;

    Position agentPos;
    Position initialAgentPos;
    Position goalPos;
    size_t currOpPosIdx;
    Actions lastOpponentAction;
    size_t traceSize;
    size_t visionGridSize;     // TODO: change these in the constructor
    size_t visionGridSideSize; // visionGridSize*2+1
    size_t agentStateSize;     // visionGridSizeSize^2

    std::vector<Position> opponentTrace;
    std::vector<Position> walls;

    // REWARDS TODO: decide on these
    float d_outOfBoundsReward;
    float d_reachedGoalReward;
    float d_killedByOpponentReward;
    float d_normalReward;

  public:
    SimState(std::string const &filename, Rewards rewards, SimStateParams simStateParams);
    void generateStateRepresentation();
    std::vector<std::vector<ImVec4>> const &getFullMazeRepr();
    void updateCanvasStepSize(ImVec2 stepSize);
    void updateCanvasBegPos(ImVec2 pos);
    void updateCanvasEndPos(ImVec2 pos);
    size_t getWidth() const;
    size_t getHeight() const;
    Position const &getSimSize() const;
    Eigen::VectorXf getStateForOpponent() const;

  public:
    void resetForNextEpisode();
    // this also moves the agent
    std::tuple<float, SimResult> computeNextStateAndReward(Actions action);
    Eigen::VectorXf getStateForAgent() const;
    float killedByOpponentReward();
    size_t getLastOpponentAction();

  private:
    std::pair<float, SimResult> updateAgentPos(Actions action);
    Position computeNewAgentPos(Actions action);
    void resetAgentPos();
    void updateOpponentPos();
};

inline size_t SimState::getWidth() const
{
    return simSize.x;
}
inline size_t SimState::getHeight() const
{
    return simSize.y;
}
inline size_t SimState::getLastOpponentAction()
{
    return static_cast<size_t>(lastOpponentAction);
}

#endif
