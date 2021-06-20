#if SHOULD_HAVE_GUI
#ifndef _INCLUDED_SIMBUILDER
#define _INCLUDED_SIMBUILDER

#include "../utilities/utilities.h"
#include "imgui.h"
#include <iostream>
#include <vector>

struct SimBuilder
{

    SimBuilder(size_t width, size_t height);
    SimBuilder() = default;
    SimBuilder &operator=(SimBuilder &&tmp);
    //    void swap(SimBuilder &other);
    std::vector<std::vector<FloatVec4>> const &getFullMazeRepr();
    void updateCanvasStepSize(FloatVec2 stepSize);
    void updateCanvasBegPos(FloatVec2 pos);
    void updateCanvasEndPos(FloatVec2 pos);
    void generateStateRepresentation();
    size_t getWidth() const;
    size_t getHeight() const;
    void drawAtPos(Position pos);
    void removeAtPos(Position pos);

    SimObject objToDraw = SimObject::NONE;

    std::vector<std::vector<SimObject>> stateRepresentation;
    std::vector<std::vector<FloatVec4>> colorRepresentation;

    FloatVec2 canvasStepSize;
    FloatVec2 canvasBegPos;
    FloatVec2 canvasEndPos;

    Position simSize;

    Position agentPos;
    Position goalPos;
    Position opponentPos;
    std::vector<Position> opponentTrace;
    std::vector<Position> walls;

    void writeToFile(std::string const &fileName);
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
#endif