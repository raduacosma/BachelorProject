#include "simState.h"
using namespace std;

std::vector <std::vector<TileStates>> SimState::getFullMazeRepr() const
{
    vector<TileStates> row{simSize.y, TileStates::EMPTY};
    return vector<vector<TileStates>>{simSize.x,row};

}

SimState::SimState(size_t width, size_t height)
:
    simSize({width,height})
{
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
