#include "monteCarloSim.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <tuple>

using namespace std;
MonteCarloSim::MonteCarloSim(SimState const &simState)
    : agentPos(simState.agentPos), simSize(simState.simSize), goalPos(simState.goalPos), traceSize(simState.traceSize),
      agentVisionGridSize(simState.agentVisionGridSize), agentVisionGridSideSize(simState.agentVisionGridSideSize),
      agentStateSize(simState.agentStateSize), opponentVisionGridSize(simState.opponentVisionGridSize),
      opponentVisionGridSideSize(simState.opponentVisionGridSideSize), opponentStateSize(simState.opponentStateSize),
      walls(simState.walls), d_outOfBoundsReward(simState.d_outOfBoundsReward),
      d_reachedGoalReward(simState.d_reachedGoalReward), d_killedByOpponentReward(simState.d_killedByOpponentReward),
      d_normalReward(simState.d_normalReward), opponentTrace(simState.currOpTrace)
{
}
// constructor from MonteCarloSim
Position MonteCarloSim::computeNewPos(Actions currAction, Position pos)
{
    // TODO: decide if x and y start from lower or higher should be fine now
    switch (currAction)
    {
        case Actions::UP:
            return { pos.x, pos.y - 1 };
        case Actions::DOWN:
            return { pos.x, pos.y + 1 };
        case Actions::LEFT:
            return { pos.x - 1, pos.y };
        case Actions::RIGHT:
            return { pos.x + 1, pos.y };
    }
    // should throw something but meh
    return { pos.x, pos.y };
}

tuple<float, SimResult> MonteCarloSim::computeNextStateAndReward(Actions action, Actions opAction)
{
    updateOpPos(opAction);
    auto [reward, canContinue] = updateAgentPos(action);
    // make sure this and hash should be updated before opponent ?? what is this

    return make_tuple(reward, canContinue);
}
void MonteCarloSim::updateOpPos(Actions opAction)
{
    Position futurePos = computeNewPos(opAction, opponentTrace.back());
    //    float reward = abs(static_cast<int>(agentPos.x - goalPos.x))+abs(static_cast<int>(agentPos.y - goalPos.y));
    if (futurePos.x < 0 or futurePos.x >= simSize.x or futurePos.y < 0 or futurePos.y >= simSize.y)
    {
        return;
    }
    if (futurePos == goalPos)
    {
        return;
    }
    for (auto const &wall : walls)
    {
        if (futurePos == wall)
        {
            return;
        }
    }
    for (auto const &tracePos : opponentTrace)
    {
        if (futurePos == tracePos)
            return;
    }

    // check walls and stuff  ??
    if (opponentTrace.size() > traceSize)
    {
        opponentTrace.pop_front();
    }
    opponentTrace.push_back(futurePos);
}
Eigen::VectorXf MonteCarloSim::getStateForAgent() const
{ // should the goal really be a vision grid?
    // also, everywhere the agent center is included for avoiding the performance cost
    // of the if and supposedly being better for 2D representations but debatable
    size_t offsetForGoal = agentStateSize * 2;
    Eigen::VectorXf agentGrid = Eigen::VectorXf::Zero(agentStateSize * 2 + 4);
    auto applyToArray = [&](Position const &pos, size_t offset)
    {
        long const rowIdx = pos.y - agentPos.y + agentVisionGridSize;
        long const colIdx = pos.x - agentPos.x + agentVisionGridSize;
        if (rowIdx >= 0 and colIdx >= 0 and rowIdx < static_cast<long>(agentVisionGridSideSize) and
            colIdx < static_cast<long>(agentVisionGridSideSize))
            agentGrid[rowIdx * agentVisionGridSideSize + colIdx + offset] = 1.0f;
    };
    for (auto const &wall : walls)
    {
        applyToArray(wall, 0);
    }
    for (auto const &opPos : opponentTrace)
    {
        applyToArray(opPos, agentStateSize);
    }
    //        applyToArray(goalPos,agentStateSize*2);
    float xDiff = static_cast<int>(goalPos.x - agentPos.x) / 5.0f;
    float yDiff = static_cast<int>(goalPos.y - agentPos.y) / 5.0f;
    if (xDiff < 0)
    {
        agentGrid[offsetForGoal] = -xDiff;
        agentGrid[offsetForGoal + 1] = 0;
    }
    else
    {
        agentGrid[offsetForGoal] = 0;
        agentGrid[offsetForGoal + 1] = xDiff;
    }
    if (yDiff < 0)
    {
        agentGrid[offsetForGoal + 2] = -yDiff;
        agentGrid[offsetForGoal + 3] = 0;
    }
    else
    {
        agentGrid[offsetForGoal + 2] = 0;
        agentGrid[offsetForGoal + 3] = yDiff;
    }
    //    agentGrid[offsetForGoal] = static_cast<int>(goalPos.x - agentPos.x) / 20.0f;
    //    agentGrid[offsetForGoal + 1] = static_cast<int>(goalPos.y - agentPos.y) / 20.0f;
    return agentGrid;
}
Eigen::VectorXf MonteCarloSim::getStateForOpponent() const
{ // should the goal really be a vision grid?
    // also, everywhere the agent center is included for avoiding the performance cost
    // of the if and supposedly being better for 2D representations but debatable
    Position currPos = opponentTrace.back();
    Eigen::VectorXf agentGrid = Eigen::VectorXf::Zero(opponentStateSize * 3);
    auto applyToArray = [&](Position const &pos, size_t offset)
    {
        long const rowIdx = pos.y - currPos.y + opponentVisionGridSize;
        long const colIdx = pos.x - currPos.x + opponentVisionGridSize;
        if (rowIdx >= 0 and colIdx >= 0 and rowIdx < static_cast<long>(opponentVisionGridSideSize) and
            colIdx < static_cast<long>(opponentVisionGridSideSize))
            agentGrid[rowIdx * opponentVisionGridSideSize + colIdx + offset] = 1.0f;
    };
    for (auto const &wall : walls)
    {
        applyToArray(wall, 0);
    }

    for (auto const &opPos : opponentTrace)
    {
        applyToArray(opPos, opponentStateSize);
    }
    applyToArray(goalPos, opponentStateSize * 2);
    return agentGrid;
}

pair<float, SimResult> MonteCarloSim::updateAgentPos(Actions action)
{
    auto futurePos = computeNewPos(action, agentPos);
    //    float reward = abs(static_cast<int>(agentPos.x - goalPos.x))+abs(static_cast<int>(agentPos.y - goalPos.y));
    if (futurePos.x < 0 or futurePos.x >= simSize.x or futurePos.y < 0 or futurePos.y >= simSize.y)
    {
        return make_pair(d_outOfBoundsReward, SimResult::CONTINUE); // false means end episode
    }
    if (futurePos == goalPos)
    {
        agentPos = futurePos; // should it go over the goal? probably does not matter at this point
        return make_pair(d_reachedGoalReward, SimResult::REACHED_GOAL);
    }
    for (auto const &wall : walls)
    {
        if (futurePos == wall)
        {
            return make_pair(d_outOfBoundsReward, SimResult::CONTINUE);
        }
    }
    for (auto const &opPos : opponentTrace)
    {
        if (futurePos == opPos)
        {
            return make_pair(d_killedByOpponentReward, SimResult::KILLED_BY_OPPONENT);
        }
    }

    // check walls and stuff ??
    agentPos = futurePos;
    return make_pair(d_normalReward, SimResult::CONTINUE);
}
