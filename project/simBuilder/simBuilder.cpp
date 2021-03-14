#include "simBuilder.h"
#include <algorithm>
using namespace std;

SimBuilder::SimBuilder(size_t width, size_t height)
    :
    simSize({width,height}),agentPos({width,height}),
      goalPos({width,height}),opponentPos({width,height}),opponentTrace(1,{width,height}),
      walls(1,{width,height})
{
}
std::vector<std::vector<TileStates>> SimBuilder::getFullMazeRepr() const
{
    vector<TileStates> row{simSize.y, TileStates::EMPTY};
    vector<vector<TileStates>> repr{simSize.x,row};

    // Since default pos's are initialized to 0, the (0,0) tile gets colored
    // even though it should not. Therefore, initialize in the constructor
    // all the empty pos's to {height,width} and then check here if they
    // are indeed out of bounds. Since this is just for the GUI, performance
    // does not really matter so it's fine
    auto assignWithBoundCheck = [&](Position pos, TileStates tileState)
    {
        if (pos.x >= getWidth() && pos.y >= getHeight())
            return;
        else
            repr[pos.x][pos.y] = tileState;
    };
    assignWithBoundCheck(agentPos,TileStates::AGENT);
    assignWithBoundCheck(goalPos,TileStates::GOAL);
    for (auto const & wall: walls)
    {
        assignWithBoundCheck(wall,TileStates::WALL);
    }
    assignWithBoundCheck(opponentPos,TileStates::OPPONENT);
    for (auto const & opPos: opponentTrace)
    {
        assignWithBoundCheck(opPos,TileStates::OPPONENT_TRACE);
    }
    return repr;
}
void SimBuilder::drawAtPos(Position pos)
{
    if(pos.x < 0 or pos.x >= getWidth() or pos.y < 0 or pos.y >= getHeight())
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
            if(std::find(walls.begin(),walls.end(), pos) == walls.end())
                walls.push_back(pos);
            break;
        case SimObject::OPPONENT:
            opponentPos = pos;
            break;
        case SimObject::OPPONENT_TRACE:
            if(std::find(opponentTrace.begin(),opponentTrace.end(), pos) == opponentTrace.end())
                opponentTrace.push_back(pos);
            break;
    }

    generateStateRepresentation();
}
void SimBuilder::removeAtPos(Position pos)
{
    if(pos.x < 0 or pos.x >= getWidth() or pos.y < 0 or pos.y >= getHeight())
        return;
    switch (objToDraw)
    {

        case SimObject::NONE:
            break;
        case SimObject::AGENT:
            if (agentPos == pos)
                agentPos = { getWidth(),getHeight() };
            break;
        case SimObject::GOAL:
            if (goalPos == pos)
                goalPos = { getWidth(),getHeight() };
            break;
        case SimObject::WALL:
            walls.erase(std::remove(walls.begin(),walls.end(), pos),walls.end());
            break;
        case SimObject::OPPONENT:
            if (opponentPos == pos)
                opponentPos = {getWidth(), getHeight()};
            break;
        case SimObject::OPPONENT_TRACE:
            opponentTrace.erase(std::remove(opponentTrace.begin(),opponentTrace.end(), pos),opponentTrace.end());
            break;
    }
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
{

}
