#ifndef _INCLUDED_SIMBUILDER
#define _INCLUDED_SIMBUILDER

#include "../utilities/utilities.h"
#include "vector"

struct SimBuilder
{
    SimBuilder(size_t width, size_t height);
//    std::vector<std::vector<TileStates>> getFullMazeRepr() const;
    size_t getWidth() const;
    size_t getHeight() const;


    pos simSize;

    pos agentPos;
    pos goalPos;
    pos opponentPos;
    std::vector<pos> walls;
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
