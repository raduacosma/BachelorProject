#ifndef _INCLUDED_SIMSTATE
#define _INCLUDED_SIMSTATE

#include "../../Eigen/Core"
#include "../utilities/utilities.h"
#include <deque>
#include <string>
#include <vector>

class Agent;

#include "actions.h"

class SimState
{
    friend class MonteCarloSim;

  private:
    std::vector<std::vector<FloatVec4>> stateRepresentation;
    FloatVec2 canvasStepSize;
    FloatVec2 canvasBegPos;
    FloatVec2 canvasEndPos;

    Position simSize;

    Position agentPos;
    Position initialAgentPos;
    Position goalPos;
    size_t currOpPosIdx;
    Actions lastOpponentAction;
    size_t traceSize;
    size_t agentVisionGridSize;
    size_t agentVisionGridSideSize; // visionGridSize*2+1
    size_t agentStateSize;          // visionGridSizeSize^2
    size_t opponentVisionGridSize;
    size_t opponentVisionGridSideSize; // visionGridSize*2+1
    size_t opponentStateSize;          // visionGridSizeSize^2

    std::vector<Position> opponentTrace;
    std::vector<Position> walls;
    std::deque<Position> currOpTrace;
    std::deque<Position> randomFluctuations;
    float randomOpCoef;

    // REWARDS
    float d_outOfBoundsReward;
    float d_reachedGoalReward;
    float d_killedByOpponentReward;
    float d_normalReward;

  public:
    SimState(std::string const &filename, Rewards rewards, SimStateParams simStateParams);
    void generateStateRepresentation();
    std::vector<std::vector<FloatVec4>> const &getFullMazeRepr();
    void updateCanvasStepSize(FloatVec2 stepSize);
    void updateCanvasBegPos(FloatVec2 pos);
    void updateCanvasEndPos(FloatVec2 pos);
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
    void createRandomFluctuations(Position const &newPos);
    static Actions computeDirection(Position const &newPos, Position const &lastPos);
    Position computeNewPos(Actions currAction, Position pos);
    bool checkPositionForOpponent(Position const &testPos);
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
inline float SimState::killedByOpponentReward()
{
    return d_killedByOpponentReward;
}

inline void SimState::updateCanvasBegPos(FloatVec2 pos)
{
    canvasBegPos = pos;
}
inline void SimState::updateCanvasEndPos(FloatVec2 pos)
{
    canvasEndPos = pos;
}
inline void SimState::updateCanvasStepSize(FloatVec2 stepSize)
{
    canvasStepSize = stepSize;
}
inline Position const &SimState::getSimSize() const
{
    return simSize;
}

#endif
