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

#include "simState.h"
#include "../agent/agent.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <tuple>

using namespace std;

SimState::SimState(std::string const &filename, Rewards rewards, SimStateParams simStateParams)
    : traceSize(simStateParams.traceSize), agentVisionGridSize(simStateParams.agentVisionGridSize),
      agentVisionGridSideSize(agentVisionGridSize * 2 + 1),
      agentStateSize(agentVisionGridSideSize * agentVisionGridSideSize),
      opponentVisionGridSize(simStateParams.opponentVisionGridSize),
      opponentVisionGridSideSize(opponentVisionGridSize * 2 + 1),
      opponentStateSize(opponentVisionGridSideSize * opponentVisionGridSideSize),
      randomOpCoef(simStateParams.randomOpCoef), d_outOfBoundsReward(rewards.outOfBoundsReward),
      d_reachedGoalReward(rewards.reachedGoalReward), d_killedByOpponentReward(rewards.killedByOpponentReward),
      d_normalReward(rewards.normalReward)
{
    ifstream in{ filename };
    if (not in)
    {
        cout << "could not open file" << endl;
        return;
    }
    std::string label;
    in >> label;
    in >> canvasStepSize.x >> canvasStepSize.y;
    in >> label;
    in >> canvasBegPos.x >> canvasBegPos.y;
    in >> label;
    in >> canvasEndPos.x >> canvasEndPos.y;
    in >> label;
    in >> simSize.x >> simSize.y;
    in >> label;
    in >> initialAgentPos.x >> initialAgentPos.y;
    in >> label;
    in >> goalPos.x >> goalPos.y;
    size_t opStartPosx, opStartPosy;
    in >> label;
    in >> opStartPosx >> opStartPosy;
    in >> label;
    size_t tracex, tracey;
    opponentTrace.push_back({ opStartPosx, opStartPosy });
    while (in >> tracex >> tracey)
    {
        opponentTrace.push_back({ tracex, tracey });
    }
    in.clear();
    in >> label;
    size_t wallsx, wallsy;
    while (in >> wallsx >> wallsy)
    {
        walls.push_back({ wallsx, wallsy });
    }
    resetForNextEpisode();
}
Position SimState::computeNewPos(Actions currAction, Position pos)
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
    return { pos.x, pos.y };
}
Position SimState::computeNewAgentPos(Actions agentAction)
{
    switch (agentAction)
    {
        case Actions::UP:
            return { agentPos.x, agentPos.y - 1 };
        case Actions::DOWN:
            return { agentPos.x, agentPos.y + 1 };
        case Actions::LEFT:
            return { agentPos.x - 1, agentPos.y };
        case Actions::RIGHT:
            return { agentPos.x + 1, agentPos.y };
    }
    return { agentPos.x, agentPos.y };
}

tuple<float, SimResult> SimState::computeNextStateAndReward(Actions action)
{
    updateOpponentPos();
    auto [reward, canContinue] = updateAgentPos(action);
    return make_tuple(reward, canContinue);
}
void SimState::updateOpponentPos()
{
    Position lastOpponentPos = currOpTrace.back();
    Position newOpponentPos;
    if (randomFluctuations.empty())
    {
        ++currOpPosIdx;
        if (currOpPosIdx == opponentTrace.size())
        {
            currOpPosIdx = 0;
        }
        newOpponentPos = opponentTrace[currOpPosIdx];

        if (globalRng.getUniReal01() < randomOpCoef)
        {
            createRandomFluctuations(newOpponentPos);
        }
    }
    else
    {
        newOpponentPos = randomFluctuations.front();
        randomFluctuations.pop_front();
    }
    if (currOpTrace.size() > traceSize)
        currOpTrace.pop_front();
    currOpTrace.push_back(newOpponentPos);
    lastOpponentAction = computeDirection(newOpponentPos, lastOpponentPos);
}
Actions SimState::computeDirection(Position const &newPos, Position const &lastPos)
{
    int xDiff = static_cast<int>(newPos.x - lastPos.x);
    int yDiff = static_cast<int>(newPos.y - lastPos.y);
    if (xDiff < 0)
        return Actions::LEFT;
    else if (xDiff > 0)
        return Actions::RIGHT;
    else if (yDiff < 0)
        return Actions::UP;
    else if (yDiff > 0)
        return Actions::DOWN;
    return Actions::DOWN;
}
void SimState::createRandomFluctuations(Position const &newPos)
{
    size_t opIdxCopy = currOpPosIdx;
    ++opIdxCopy;
    if (opIdxCopy == opponentTrace.size())
    {
        opIdxCopy = 0;
    }
    auto updateFluctuationsWithDirections = [&](Actions first, Actions second) -> bool
    {
        Position afterFirst = computeNewPos(first, newPos);
        Position afterSecond = computeNewPos(second, afterFirst);
        if (checkPositionForOpponent(afterFirst) and checkPositionForOpponent(afterSecond))
        {
            randomFluctuations.push_back(afterFirst);
            randomFluctuations.push_back(afterSecond);
            return true;
        }
        return false;
    };
    Actions direction = computeDirection(opponentTrace[opIdxCopy], newPos);
    switch (direction)
    {
        case Actions::LEFT:
            if (not updateFluctuationsWithDirections(Actions::DOWN, Actions::LEFT))
                updateFluctuationsWithDirections(Actions::UP, Actions::LEFT);
            break;
        case Actions::RIGHT:
            if (not updateFluctuationsWithDirections(Actions::DOWN, Actions::RIGHT))
                updateFluctuationsWithDirections(Actions::UP, Actions::RIGHT);
            break;
        case Actions::UP:
            if (not updateFluctuationsWithDirections(Actions::LEFT, Actions::UP))
                updateFluctuationsWithDirections(Actions::RIGHT, Actions::UP);
            break;
        case Actions::DOWN:
            if (not updateFluctuationsWithDirections(Actions::LEFT, Actions::DOWN))
                updateFluctuationsWithDirections(Actions::RIGHT, Actions::DOWN);
            break;
    }
}

Eigen::VectorXf SimState::getStateForAgent() const
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

    for (auto const &opPos : currOpTrace)
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
Eigen::VectorXf SimState::getStateForOpponent() const
{
    Eigen::VectorXf agentGrid = Eigen::VectorXf::Zero(opponentStateSize * 3);
    Position opPosNow = currOpTrace.back();
    auto applyToArray = [&](Position const &pos, size_t offset)
    {
        long const rowIdx = pos.y - opPosNow.y + opponentVisionGridSize;
        long const colIdx = pos.x - opPosNow.x + opponentVisionGridSize;
        if (rowIdx >= 0 and colIdx >= 0 and rowIdx < static_cast<long>(opponentVisionGridSideSize) and
            colIdx < static_cast<long>(opponentVisionGridSideSize))
            agentGrid[rowIdx * opponentVisionGridSideSize + colIdx + offset] = 1.0f;
    };
    for (auto const &wall : walls)
    {
        applyToArray(wall, 0);
    }
    for (auto const &opPos : currOpTrace)
    {
        applyToArray(opPos, opponentStateSize);
    }
    applyToArray(goalPos, opponentStateSize * 2);
    return agentGrid;
}
void SimState::resetAgentPos()
{
    std::uniform_int_distribution<> distr{ -3, 3 };
    auto &rngEngine = globalRng.getRngEngine();
    agentPos = { initialAgentPos.x, initialAgentPos.y + distr(rngEngine) };
    //    agentPos = initialAgentPos;
}

void SimState::resetForNextEpisode()
{
    std::uniform_int_distribution<> distr(0, opponentTrace.size() - 1);
    currOpPosIdx = distr(globalRng.getRngEngine());
    size_t opLength = traceSize + 1;
    // be careful, we can't do the >-1 check due to size_t and this should
    // stop after 0 but if something is wrong good to check this
    randomFluctuations.resize(0);
    currOpTrace.resize(0);
    for (size_t idx = currOpPosIdx + 1; idx-- > 0 and opLength;)
    {
        currOpTrace.push_front(opponentTrace[idx]);
        --opLength;
    }
    for (size_t idx = opponentTrace.size(); idx-- > 0 and opLength;)
    {
        currOpTrace.push_front(opponentTrace[idx]);
        --opLength;
    }
    resetAgentPos();
}
bool SimState::checkPositionForOpponent(Position const &futurePos)
{
    //    float reward = abs(static_cast<int>(agentPos.x - goalPos.x))+abs(static_cast<int>(agentPos.y - goalPos.y));
    if (futurePos.x < 0 or futurePos.x >= simSize.x or futurePos.y < 0 or futurePos.y >= simSize.y)
    {
        return false;
    }
    if (futurePos == goalPos)
    {
        return false;
    }
    for (auto const &wall : walls)
    {
        if (futurePos == wall)
        {
            return false;
        }
    }
    for (auto const &tracePos : opponentTrace)
    {
        if (futurePos == tracePos)
            return false;
    }
    for (auto const &tracePos : currOpTrace)
    {
        if (futurePos == tracePos)
            return false;
    }
    return true;
}
pair<float, SimResult> SimState::updateAgentPos(Actions action)
{
    auto futurePos = computeNewAgentPos(action);
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
    for (auto const &opPos : currOpTrace)
    {
        if (futurePos == opPos)
        {
            return make_pair(d_killedByOpponentReward, SimResult::KILLED_BY_OPPONENT);
        }
    }
    agentPos = futurePos;
    return make_pair(d_normalReward, SimResult::CONTINUE);
}

std::vector<std::vector<FloatVec4>> const &SimState::getFullMazeRepr()
{
    generateStateRepresentation();
    return stateRepresentation;
}

void SimState::generateStateRepresentation()
{
    vector<FloatVec4> row{ simSize.y, { 255, 255, 255, 255 } };
    vector<vector<FloatVec4>> repr{ simSize.x, row };

    // Since default pos's are initialized to 0, the (0,0) tile gets colored
    // even though it should not. Therefore, initialize in the constructor
    // all the empty pos's to {height,width} and then check here if they
    // are indeed out of bounds. Since this is just for the GUI, performance
    // does not really matter
    auto assignWithBoundCheck = [&](Position pos, FloatVec4 color)
    {
        if (pos.x >= getWidth() || pos.y >= getHeight() || pos.x < 0 || pos.y < 0)
            return;
        else
            repr[pos.x][pos.y] = color;
    };
    FloatVec4 agentColor = { 0, 0, 255, 255 };
    FloatVec4 goalColor = { 0, 128, 0, 255 };
    FloatVec4 wallColor = { 128, 128, 128, 255 };
    FloatVec4 opponentColor = { 255, 0, 0, 255 };
    FloatVec4 opponentTraceColor = { 178, 34, 34, 255 };
    FloatVec4 agentViewColor = { 135, 206, 235, 255 };
    FloatVec4 opponentViewColor = { 202, 119, 119, 255 };
    assignWithBoundCheck(agentPos, agentColor);
    assignWithBoundCheck(goalPos, goalColor);
    for (auto const &wall : walls)
    {
        assignWithBoundCheck(wall, wallColor);
    }

    //    assignWithBoundCheck(opponentPos,SimObject::OPPONENT);
    FloatVec4 color = opponentColor;
    float factor = 0.075f;
    for (auto const &item : currOpTrace)
    {
        //        color =
        //        {(255.0f-color.x)*factor+color.x,(255.0f-color.y)*factor+color.y,(255.0f-color.z)*factor+color.z,255};
        ////        color = {(color.x+255.0f)/2,(color.y+255.0f)/2,(color.z+255.0f)/2,color.w};
        assignWithBoundCheck(item, opponentTraceColor);
    }
    assignWithBoundCheck(currOpTrace.back(), opponentColor);
    auto applyViewColor = [&](Position pos, FloatVec4 color, size_t visionGridSize)
    {
        for (size_t i = 0; i < simSize.x; ++i)
            for (size_t j = 0; j < simSize.y; ++j)
            {
                if (abs(static_cast<int>(i - pos.x)) <= visionGridSize and
                    abs(static_cast<int>(j - pos.y)) <= visionGridSize)
                {
                    repr[i][j] = { (repr[i][j].x + color.x) / 2, (repr[i][j].y + color.y) / 2,
                                   (repr[i][j].z + color.z) / 2, 255 };
                }
            }
    };
    applyViewColor(agentPos, agentViewColor, agentVisionGridSize);
    applyViewColor(currOpTrace.back(), opponentViewColor, opponentVisionGridSize);

    stateRepresentation = move(repr);
}
