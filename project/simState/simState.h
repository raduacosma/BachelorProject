#ifndef _INCLUDED_SIMSTATE
#define _INCLUDED_SIMSTATE

#include <vector>
#include "../utilities/utilities.h"
#include "imgui.h"
#include <string>
#include "../Eigen/Core"

class Agent;

#include "actions.h"

class SimState
{
    public:
    SimState() = default;
    SimState(std::string const &filename);
    void generateStateRepresentation();
    std::vector<std::vector<ImVec4>> const & getFullMazeRepr();
    void updateCanvasStepSize(ImVec2 stepSize);
    void updateCanvasBegPos(ImVec2 pos);
    void updateCanvasEndPos(ImVec2 pos);
    size_t getWidth() const;
    size_t getHeight() const;


    private:

    std::vector<std::vector<ImVec4>> stateRepresentation;
    ImVec2 canvasStepSize;
    ImVec2 canvasBegPos;
    ImVec2 canvasEndPos;

    Position simSize;

    public:
    Position const &getSimSize() const;
    private:
    Position agentPos;
    Position initialAgentPos;
    Position goalPos;
    size_t currOpPosIdx;
    size_t traceSize = 5;
    size_t visionGridSize = 2;  // TODO: change these in the constructor
    size_t visionGridSideSize = 5;  // visionGridSize*2+1
    size_t agentStateSize = 25;     // visionGridSizeSize^2

    std::vector<Position> opponentTrace;
    std::vector<Position> walls;

    Actions currAction;

    // REWARDS TODO: decide on these
    float d_outOfBoundsReward;
    float d_reachedGoalReward;
    float d_killedByOpponentReward;
    float d_normalReward;

    public:
    void resetForNextEpisode();
    // this also moves the agent
    std::tuple<float, SimResult> computeNextStateAndReward(Actions action);
    Eigen::VectorXf getStateForAgent() const;
    float killedByOpponentReward();

    private:
    std::pair<float, SimResult> updateAgentPos();
    Position computeNewAgentPos();
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



#endif
