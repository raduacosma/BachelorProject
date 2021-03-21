#include "../agent/agent.h"
#include "simState.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <tuple>
#include "../random/random.h"
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

tuple<double, size_t, bool> SimState::computeNextStateAndReward(Actions action)
{
    currAction = action;

    auto [reward, canContinue] = updateAgentPos();
    // make sure this should be updated before opponent
    size_t hash = mazeStateHash();
    bool returnContinue;
    switch (canContinue)
    {

        case SimResult::CONTINUE:
            returnContinue = true;
            updateOpponentPos();
            break;
        case SimResult::REACHED_GOAL:
            returnContinue = true;
            break;
        case SimResult::KILLED_BY_OPPONENT:
            returnContinue = false;
            resetForNextEpisode();
            break;
    }


    return make_tuple(reward, hash, returnContinue);
}
void SimState::updateOpponentPos()
{
    ++currOpPosIdx;
    if(currOpPosIdx == opponentTrace.size())
    {
        currOpPosIdx = 0;
    }
    cout<<currOpPosIdx<<endl;
}


double SimState::killedByOpponentReward()
{
    return d_killedByOpponentReward;
}
size_t SimState::mazeStateHash() const
{

    size_t stateHash = agentPos.x * simSize.y + agentPos.y;
    return stateHash;
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

void SimState::sendNrStatesToAgent()
{
    // will need to be refactored soon
//    agent->stateSpaceSize(simSize.y * simSize.x);
}

pair<double, SimResult> SimState::updateAgentPos()
{

    auto futurePos = computeNewAgentPos();
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

std::vector <std::vector<SimObject>> const & SimState::getFullMazeRepr()
{
    generateStateRepresentation();
    return stateRepresentation;
}

SimState::SimState(std::string const &filename)
:
 d_outOfBoundsReward(-1), d_reachedGoalReward(1),
      d_killedByOpponentReward(-100), d_normalReward(-1)
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
    correctState = true;

//    sendNrStatesToAgent();
    resetForNextEpisode();
}

void SimState::generateStateRepresentation()
{
    vector<SimObject> row{simSize.y, SimObject::NONE};
    vector<vector<SimObject>> repr{simSize.x,row};

    // Since default pos's are initialized to 0, the (0,0) tile gets colored
    // even though it should not. Therefore, initialize in the constructor
    // all the empty pos's to {height,width} and then check here if they
    // are indeed out of bounds. Since this is just for the GUI, performance
    // does not really matter so it's fine
    auto assignWithBoundCheck = [&](Position pos, SimObject tileState)
    {
      if (pos.x >= getWidth() || pos.y >= getHeight() || pos.x < 0 || pos.y < 0)
          return;
      else
          repr[pos.x][pos.y] = tileState;
    };

    assignWithBoundCheck(agentPos,SimObject::AGENT);
    assignWithBoundCheck(goalPos,SimObject::GOAL);
    for (auto const & wall: walls)
    {
        assignWithBoundCheck(wall,SimObject::WALL);
    }
//    assignWithBoundCheck(opponentPos,SimObject::OPPONENT);
    assignWithBoundCheck(opponentTrace[currOpPosIdx],SimObject::OPPONENT);
    size_t opLength = traceSize;  // replace with trace size
    // be careful, we can't do the >-1 check due to size_t and this should
    // stop after 0 but if something is wrong good to check this

    for (size_t idx = currOpPosIdx; idx-- > 0 and opLength;)
    {
        assignWithBoundCheck(opponentTrace[idx], SimObject::OPPONENT_TRACE);
        --opLength;
    }
    for (size_t idx = opponentTrace.size(); idx-- > 0 and opLength;)
    {
        assignWithBoundCheck(opponentTrace[idx], SimObject::OPPONENT_TRACE);
        --opLength;
    }
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
