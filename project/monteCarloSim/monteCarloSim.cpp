/*
     Copyright (C) 2021  Radu Alexandru Cosma

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

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
Position MonteCarloSim::computeNewPos(Actions currAction, Position pos)
{
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
    // should never reach this
    return { pos.x, pos.y };
}

tuple<float, SimResult> MonteCarloSim::computeNextStateAndReward(Actions action, Actions opAction)
{
    updateOpPos(opAction);
    auto [reward, canContinue] = updateAgentPos(action);
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
    if (opponentTrace.size() > traceSize)
    {
        opponentTrace.pop_front();
    }
    opponentTrace.push_back(futurePos);
}
Eigen::VectorXf MonteCarloSim::getStateForAgent() const
{
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
    float xDiff = (static_cast<int>(goalPos.x) - static_cast<int>(agentPos.x)) / 5.0f;
    float yDiff = (static_cast<int>(goalPos.y) - static_cast<int>(agentPos.y)) / 5.0f;
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
    return agentGrid;
}
Eigen::VectorXf MonteCarloSim::getStateForOpponent() const
{
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
        return make_pair(d_outOfBoundsReward, SimResult::CONTINUE);
    }
    if (futurePos == goalPos)
    {
        agentPos = futurePos;
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

    agentPos = futurePos;
    return make_pair(d_normalReward, SimResult::CONTINUE);
}
