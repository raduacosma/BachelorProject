#ifndef _INCLUDED_SIMSTATE
#define _INCLUDED_SIMSTATE

#include <vector>
#include "../utilities/utilities.h"

enum class TileStates
{
    AGENT,
    OPPONENT,
    AGENT_TRACE,
    OPPONENT_TRACE,
    WALL,
    GOAL,
    EMPTY
};

class SimState
{
    public:
    SimState(size_t width, size_t height);
    std::vector<std::vector<TileStates>> getFullMazeRepr() const;
    size_t getWidth() const;
    size_t getHeight() const;
    private:
    pos simSize;

    pos agentPos;
    pos goalPos;
    std::vector<pos> currOppPosHist;
    std::vector<pos> walls;




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
