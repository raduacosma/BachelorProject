#include "../agent/agent.h"
#include "simState.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <tuple>


using namespace std;



Position SimState::computeNewAgentPos()
{
    // TODO: decide if x and y start from lower or higher
    switch (currAction)
    {
        case Actions::UP:
            return { agentPos.x, agentPos.y -1};
        case Actions::DOWN:
            return { agentPos.x, agentPos.y + 1};
        case Actions::LEFT:
            return { agentPos.x - 1, agentPos.y };
        case Actions::RIGHT:
            return { agentPos.x + 1, agentPos.y };
    }
    // should throw something but meh
    return { agentPos.x, agentPos.y };
}

tuple<float, SimResult> SimState::computeNextStateAndReward(Actions action)
{
    currAction = action;

    auto [reward, canContinue] = updateAgentPos();
    // make sure this and hash should be updated before opponent
    if(canContinue == SimResult::CONTINUE)
    {
        updateOpponentPos();
    }

    return make_tuple(reward, canContinue);
}
void SimState::updateOpponentPos()
{
    ++currOpPosIdx;
    if(currOpPosIdx == opponentTrace.size())
    {
        currOpPosIdx = 0;
    }
}


float SimState::killedByOpponentReward()
{
    return d_killedByOpponentReward;
}
Eigen::VectorXf SimState::getStateForAgent() const
{   // should the goal really be a vision grid?
    // also, everywhere the agent center is included for avoiding the performance cost
    // of the if and supposedly being better for 2D representations but debatable
    Eigen::VectorXf agentGrid = Eigen::VectorXf::Zero(agentStateSize*3);
    auto applyToArray = [&](Position const &pos, size_t offset)
    {
        long const rowIdx = pos.y-agentPos.y+visionGridSize;
        long const colIdx = pos.x-agentPos.x+visionGridSize;
        if(rowIdx >= 0 and colIdx >= 0 and rowIdx < static_cast<long>(visionGridSideSize) and colIdx < static_cast<long>(visionGridSideSize))
            agentGrid[rowIdx*visionGridSideSize+colIdx+offset] = 1.0f;
    };
    for (auto const &wall:walls)
    {
        applyToArray(wall,0);
    }
    applyToArray(opponentTrace[currOpPosIdx],agentStateSize);
    size_t opLength = traceSize;  // replace with trace size
    // be careful, we can't do the >-1 check due to size_t and this should
    // stop after 0 but if something is wrong good to check this

    for (size_t idx = currOpPosIdx; idx-- > 0 and opLength;)
    {
        applyToArray(opponentTrace[idx], agentStateSize);
        --opLength;
    }
    for (size_t idx = opponentTrace.size(); idx-- > 0 and opLength;)
    {
        applyToArray(opponentTrace[idx], agentStateSize);
        --opLength;
    }
    applyToArray(goalPos,agentStateSize*2);
    return agentGrid;
}

void SimState::resetAgentPos()
{
    agentPos = initialAgentPos;
}

void SimState::resetForNextEpisode()
{
    currOpPosIdx = 0;
    // reset the state but needs to feed back in the cycle I guess
    resetAgentPos();
}


pair<float, SimResult> SimState::updateAgentPos()
{
    auto futurePos = computeNewAgentPos();
//    float reward = abs(static_cast<int>(agentPos.x - goalPos.x))+abs(static_cast<int>(agentPos.y - goalPos.y));
    if (futurePos.x < 0 or futurePos.x >= simSize.x or futurePos.y < 0 or
        futurePos.y >= simSize.y)
    {
        return make_pair(d_outOfBoundsReward, SimResult::CONTINUE); // false means end episode
    }
    if (futurePos == goalPos)
    {
        agentPos = futurePos;
        return make_pair(d_reachedGoalReward, SimResult::REACHED_GOAL);
    }
    for (auto const &wall:walls)
    {
        if (futurePos == wall)
        {
            return make_pair(d_outOfBoundsReward, SimResult::CONTINUE);
        }
    }
    size_t opLength = traceSize + 1;  // replace with trace size
    // be careful, we can't do the >-1 check due to size_t and this should
    // stop after 0 but if something is wrong good to check this
    for (size_t idx = currOpPosIdx + 1; idx-- > 0 and opLength;)
    {
        if (futurePos == opponentTrace[idx])
        {
            return make_pair(d_killedByOpponentReward, SimResult::KILLED_BY_OPPONENT);
        }
        --opLength;
    }
    for (size_t idx = opponentTrace.size(); idx-- > 0 and opLength;)
    {
        if (futurePos == opponentTrace[idx])
        {
            return make_pair(d_killedByOpponentReward, SimResult::KILLED_BY_OPPONENT);
        }
        --opLength;
    }
    // check walls and stuff
    agentPos = futurePos;
    return make_pair(d_normalReward, SimResult::CONTINUE);
}

std::vector <std::vector<ImVec4>> const & SimState::getFullMazeRepr()
{
    generateStateRepresentation();
    return stateRepresentation;
}

SimState::SimState(std::string const &filename)
:
 d_outOfBoundsReward(-0.02), d_reachedGoalReward(1),
      d_killedByOpponentReward(-10), d_normalReward(-0.01)
{
    ifstream in{filename};
    if(not in)
    {
        cout << "could not open file" << endl;
        return;
    }
    std::string label;
    in >> label;
    in >> canvasStepSize.x>>canvasStepSize.y;
    in >> label;
    in >> canvasBegPos.x>>canvasBegPos.y;
    in >> label;
    in >> canvasEndPos.x>>canvasEndPos.y;
    in >> label;
    in >> simSize.x>>simSize.y;
    in >> label;
    in >> initialAgentPos.x>>initialAgentPos.y;
    in >> label;
    in >> goalPos.x>>goalPos.y;
    size_t opStartPosx,opStartPosy;
    in >> label;
    in >> opStartPosx>>opStartPosy;
    in >> label;
    size_t tracex,tracey;
    opponentTrace.push_back({opStartPosx,opStartPosy});
    while (in >> tracex>>tracey)
    {
        opponentTrace.push_back({tracex,tracey});
    }
    in.clear();
    in >> label;
    size_t wallsx,wallsy;
    while (in >> wallsx>>wallsy)
    {
        walls.push_back({wallsx,wallsy});
    }

//    sendNrStatesToAgent();
    resetForNextEpisode();
}

void SimState::generateStateRepresentation()
{
    vector<ImVec4> row{simSize.y, { 211,211,211,255 } };
    vector<vector<ImVec4>> repr{simSize.x,row};

    // Since default pos's are initialized to 0, the (0,0) tile gets colored
    // even though it should not. Therefore, initialize in the constructor
    // all the empty pos's to {height,width} and then check here if they
    // are indeed out of bounds. Since this is just for the GUI, performance
    // does not really matter so it's fine
    auto assignWithBoundCheck = [&](Position pos, ImVec4 color)
    {
      if (pos.x >= getWidth() || pos.y >= getHeight() || pos.x < 0 || pos.y < 0)
          return;
      else
          repr[pos.x][pos.y] = color;
    };
    ImVec4 agentColor = {0,0,255,255};
    ImVec4 goalColor = {0,128,0,255};
    ImVec4 wallColor = {128,128,128,255};
    ImVec4 opponentColor = {255,0,0,255};
    ImVec4 opponentTraceColor = {243,122,122,255};
    ImVec4 agentViewColor = {135,206,235,255};
    ImVec4 opponentViewColor = {202,119,119,255};
    assignWithBoundCheck(agentPos,agentColor);
    assignWithBoundCheck(goalPos,goalColor);
    for (auto const & wall: walls)
    {
        assignWithBoundCheck(wall,wallColor);
    }
//    assignWithBoundCheck(opponentPos,SimObject::OPPONENT);
    assignWithBoundCheck(opponentTrace[currOpPosIdx],opponentColor);
    size_t opLength = traceSize;  // replace with trace size
    // be careful, we can't do the >-1 check due to size_t and this should
    // stop after 0 but if something is wrong good to check this

    for (size_t idx = currOpPosIdx; idx-- > 0 and opLength;)
    {
        assignWithBoundCheck(opponentTrace[idx], opponentTraceColor);
        --opLength;
    }
    for (size_t idx = opponentTrace.size(); idx-- > 0 and opLength;)
    {
        assignWithBoundCheck(opponentTrace[idx], opponentTraceColor);
        --opLength;
    }
    auto applyViewColor = [&](Position pos, ImVec4 color)
    {
        for (size_t i = 0; i < simSize.x; ++i)
            for (size_t j = 0; j < simSize.y; ++j)
            {
                if (abs(static_cast<int>(i - pos.x)) <=
                        visionGridSize and
                    abs(static_cast<int>(j - pos.y)) <= visionGridSize)
                {
                    repr[i][j] = { (repr[i][j].x + color.x) / 2,
                                   (repr[i][j].y + color.y) / 2,
                                   (repr[i][j].z + color.z) / 2,
                                   255 };
                }
            }
    };
    applyViewColor(agentPos, agentViewColor);
    applyViewColor(opponentTrace[currOpPosIdx], opponentViewColor);

    stateRepresentation = move(repr);
}

void SimState::updateCanvasBegPos(ImVec2 pos)
{
    canvasBegPos = pos;
}
void SimState::updateCanvasEndPos(ImVec2 pos)
{
    canvasEndPos = pos;
}
void SimState::updateCanvasStepSize(ImVec2 stepSize)
{
    canvasStepSize = stepSize;
}
Position const &SimState::getSimSize() const
{
    return simSize;
}
