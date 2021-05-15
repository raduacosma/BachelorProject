#include "simBuilder.h"
#include <algorithm>
#include <fstream>
using namespace std;

SimBuilder::SimBuilder(size_t width, size_t height)
    : simSize({ width, height }), agentPos({ width, height }), goalPos({ width, height }),
      opponentPos({ width, height }), opponentTrace(1, { width, height }), walls(1, { width, height })
{
    generateStateRepresentation();
}
std::vector<std::vector<ImVec4>> const &SimBuilder::getFullMazeRepr()
{ // maybe have generateStateRepr here and remove the move constr along
    // with refactoring main to use uniquePtr for SimBuilder?
    generateStateRepresentation();
    vector<ImVec4> row{ simSize.y, { 255, 255, 255, 255 } };
    vector<vector<ImVec4>> repr{ simSize.x, row };

    for (size_t i = 0; i < simSize.x; ++i)
        for (size_t j = 0; j < simSize.y; ++j)
        {
            switch (stateRepresentation[i][j])
            {

                case SimObject::NONE:
                    repr[i][j] = { 211, 211, 211, 255 };
                    break;
                case SimObject::AGENT:
                    repr[i][j] = { 0, 0, 255, 255 };
                    break;
                case SimObject::GOAL:
                    repr[i][j] = { 0, 128, 0, 255 };
                    break;
                case SimObject::WALL:
                    repr[i][j] = { 128, 128, 128, 255 };
                    break;
                case SimObject::OPPONENT:
                    repr[i][j] = { 255, 0, 0, 255 };
                    break;
                case SimObject::OPPONENT_TRACE:
                    repr[i][j] = { 243, 122, 122, 255 };
                    break;
            }
        }
    colorRepresentation = move(repr);
    return colorRepresentation;
}
void SimBuilder::drawAtPos(Position pos)
{
    if (pos.x < 0 or pos.x >= getWidth() or pos.y < 0 or pos.y >= getHeight())
        return;
    if (stateRepresentation[pos.x][pos.y] != SimObject::NONE)
        return;
    switch (objToDraw)
    {

        case SimObject::NONE:
            break;
        case SimObject::AGENT:
            agentPos = pos;
            break;
        case SimObject::GOAL:
            goalPos = pos;
            break;
        case SimObject::WALL:
            if (std::find(walls.begin(), walls.end(), pos) == walls.end())
                walls.push_back(pos);
            break;
        case SimObject::OPPONENT:
            opponentPos = pos;
            break;
        case SimObject::OPPONENT_TRACE:
            if (std::find(opponentTrace.begin(), opponentTrace.end(), pos) == opponentTrace.end())
                opponentTrace.push_back(pos);
            break;
    }

    generateStateRepresentation();
}
void SimBuilder::removeAtPos(Position pos)
{
    if (pos.x < 0 or pos.x >= getWidth() or pos.y < 0 or pos.y >= getHeight())
        return;
    switch (objToDraw)
    {

        case SimObject::NONE:
            break;
        case SimObject::AGENT:
            if (agentPos == pos)
                agentPos = { getWidth(), getHeight() };
            break;
        case SimObject::GOAL:
            if (goalPos == pos)
                goalPos = { getWidth(), getHeight() };
            break;
        case SimObject::WALL:
            walls.erase(std::remove(walls.begin(), walls.end(), pos), walls.end());
            break;
        case SimObject::OPPONENT:
            if (opponentPos == pos)
                opponentPos = { getWidth(), getHeight() };
            break;
        case SimObject::OPPONENT_TRACE:
            opponentTrace.erase(std::remove(opponentTrace.begin(), opponentTrace.end(), pos), opponentTrace.end());
            break;
    }
    generateStateRepresentation();
}
void SimBuilder::updateCanvasBegPos(ImVec2 pos)
{
    canvasBegPos = pos;
}
void SimBuilder::updateCanvasStepSize(ImVec2 stepSize)
{
    canvasStepSize = stepSize;
}
void SimBuilder::updateCanvasEndPos(ImVec2 pos)
{
    canvasEndPos = pos;
}
void SimBuilder::generateStateRepresentation()
{ // TODO: make this more like SimState
    vector<SimObject> row{ simSize.y, SimObject::NONE };
    vector<vector<SimObject>> repr{ simSize.x, row };

    // Since default pos's are initialized to 0, the (0,0) tile gets colored
    // even though it should not. Therefore, initialize in the constructor
    // all the empty pos's to {height,width} and then check here if they
    // are indeed out of bounds. Since this is just for the GUI, performance
    // does not really matter so it's fine
    auto assignWithBoundCheck = [&](Position pos, SimObject tileState)
    {
        if (pos.x >= getWidth() && pos.y >= getHeight())
            return;
        else
            repr[pos.x][pos.y] = tileState;
    };
    assignWithBoundCheck(agentPos, SimObject::AGENT);
    assignWithBoundCheck(goalPos, SimObject::GOAL);
    for (auto const &wall : walls)
    {
        assignWithBoundCheck(wall, SimObject::WALL);
    }
    assignWithBoundCheck(opponentPos, SimObject::OPPONENT);
    for (auto const &opPos : opponentTrace)
    {
        assignWithBoundCheck(opPos, SimObject::OPPONENT_TRACE);
    }
    stateRepresentation = move(repr);
}
void SimBuilder::writeToFile(string const &fileName)
{
    ofstream out{ fileName };
    out << "canvasStepSize" << '\n'
        << canvasStepSize.x << " " << canvasStepSize.y << '\n'
        << "canvasBegPos" << '\n'
        << canvasBegPos.x << " " << canvasBegPos.y << '\n'
        << "canvasEndPos" << '\n'
        << canvasEndPos.x << " " << canvasEndPos.y << '\n'
        << "simSize" << '\n'
        << simSize.x << " " << simSize.y << '\n'
        << "agentPos" << '\n'
        << agentPos.x << " " << agentPos.y << '\n'
        << "goalPos" << '\n'
        << goalPos.x << " " << goalPos.y << '\n'
        << "opponentPos" << '\n'
        << opponentPos.x << " " << opponentPos.y << '\n';
    out << "opponentTrace" << '\n';
    for (auto const &opTr : opponentTrace)
    {
        if (opTr.x < 0 || opTr.x >= simSize.x || opTr.y < 0 || opTr.y >= simSize.y)
            continue;
        out << opTr.x << " " << opTr.y << '\n';
    }

    out << "walls" << '\n';
    for (auto const &wall : walls)
    {
        if (wall.x < 0 || wall.x >= simSize.x || wall.y < 0 || wall.y >= simSize.y)
            continue;
        out << wall.x << " " << wall.y << '\n';
    }
}
