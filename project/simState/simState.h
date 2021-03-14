#ifndef _INCLUDED_SIMSTATE
#define _INCLUDED_SIMSTATE

#include <vector>
#include "../utilities/utilities.h"
#include "imgui.h"


class SimState
{
    public:
    SimState(size_t width, size_t height);
    std::vector<std::vector<TileStates>> getFullMazeRepr() const;
    void updateCanvasStepSize(ImVec2 stepSize);
    void updateCanvasBegPos(ImVec2 pos);
    void updateCanvasEndPos(ImVec2 pos);
    size_t getWidth() const;
    size_t getHeight() const;
    private:

    ImVec2 canvasStepSize;
    ImVec2 canvasBegPos;
    ImVec2 canvasEndPos;

    Position simSize;

    Position agentPos;
    Position goalPos;
    std::vector<Position> currOppPosHist;
    std::vector<Position> walls;




};

inline size_t SimState::getWidth() const
{
    return simSize.x;
}
inline size_t SimState::getHeight() const
{
    return simSize.y;
}


#endif
