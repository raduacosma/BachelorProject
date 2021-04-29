#ifndef _INCLUDED_UTILITIES
#define _INCLUDED_UTILITIES

#include <cstddef>
#include <ostream>

// TODO: when removing enum, make sure that agent and opponent view are there
// and remove agent_trace
enum class SimObject
{
    NONE,
    AGENT,
    GOAL,
    WALL,
    OPPONENT,
    OPPONENT_TRACE,
};

enum class SimResult
{
    CONTINUE,
    REACHED_GOAL,
    KILLED_BY_OPPONENT
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

inline std::ostream &operator<<(std::ostream &out, Position pos)
{
    out << pos.x << " " << pos.y << std::endl;
    return out;
}


#endif
