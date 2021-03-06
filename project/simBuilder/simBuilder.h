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