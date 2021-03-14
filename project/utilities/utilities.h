#ifndef _INCLUDED_UTILITIES
#define _INCLUDED_UTILITIES

#include <cstddef>
// TODO: when removing enum, make sure that agent and opponent view are there
// and remove agent_trace
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

struct Position
{
    size_t x;
    size_t y;
};

struct CanvasPos
{
    double x;
    double y;
};

inline bool operator==(Position lhs, Position rhs)
{
    return lhs.x==rhs.x and lhs.y==rhs.y;
}

#endif
