#ifndef _INCLUDED_SIMBUILDER
#define _INCLUDED_SIMBUILDER

#include "../utilities/utilities.h"
#include <vector>
#include <iostream>
#include "imgui.h"

enum class SimObject
{
    NONE,
    AGENT,
    GOAL,
    WALL,
    OPPONENT,
    OPPONENT_TRACE,
};
struct SimBuilder
{
    SimObject objToDraw = SimObject::NONE;
    SimBuilder(size_t width, size_t height);
    std::vector<std::vector<TileStates>> getFullMazeRepr() const;
    void updateCanvasStepSize(ImVec2 stepSize);
    void updateCanvasBegPos(ImVec2 pos);
    void updateCanvasEndPos(ImVec2 pos);
    void generateStateRepresentation();
    size_t getWidth() const;
    size_t getHeight() const;

    std::vector<std::vector<SimObject>> stateRepresentation;
    ImVec2 canvasStepSize;
    ImVec2 canvasBegPos;
    ImVec2 canvasEndPos;
    Position simSize;

    Position agentPos;
    Position goalPos;
    Position opponentPos;
    std::vector<Position> opponentTrace;
    std::vector<Position> walls;
    void drawAtPos(Position pos);
    void removeAtPos(Position pos);
};


inline size_t SimBuilder::getWidth() const
{
    return simSize.x;
}
inline size_t SimBuilder::getHeight() const
{
    return simSize.y;
}

#endif
