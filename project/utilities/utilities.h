#ifndef _INCLUDED_UTILITIES
#define _INCLUDED_UTILITIES

#include <cstddef>
#include <ostream>
#include <vector>
// TODO: when removing enum, make sure that agent and opponent view are there
// and remove agent_trace
// 2D vector (often used to store positions or sizes)
struct FloatVec2
{
    float                                   x, y;
    FloatVec2()                                { x = y = 0.0f; }
    FloatVec2(float _x, float _y)              { x = _x; y = _y; }
    float  operator[] (size_t idx) const    { return (&x)[idx]; }    // We very rarely use this [] operator, the assert overhead is fine.
    float& operator[] (size_t idx)          { return (&x)[idx]; }    // We very rarely use this [] operator, the assert overhead is fine.
#ifdef IM_VEC2_CLASS_EXTRA
    IM_VEC2_CLASS_EXTRA     // Define additional constructors and implicit cast operators in imconfig.h to convert back and forth between your math types and ImVec2.
#endif
};

// 4D vector (often used to store floating-point colors)
struct FloatVec4
{
    float                                           x, y, z, w;
    FloatVec4()                                        { x = y = z = w = 0.0f; }
    FloatVec4(float _x, float _y, float _z, float _w)  { x = _x; y = _y; z = _z; w = _w; }
#ifdef IM_VEC4_CLASS_EXTRA
    IM_VEC4_CLASS_EXTRA     // Define additional constructors and implicit cast operators in imconfig.h to convert back and forth between your math types and ImVec4.
#endif
};
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
enum class ActivationFunction
{
    LINEAR,
    SIGMOID,
    SOFTMAX
};
struct ExpReplayParams
{
    size_t cSwapPeriod;
    size_t miniBatchSize;
    size_t sizeExperience;
};

struct AgentMonteCarloParams
{
    size_t maxNrSteps;
    size_t nrRollouts;
};
struct MLPParams
{
    std::vector<size_t> sizes;
    float learningRate;
    float regParam;
    ActivationFunction outputActivationFunc;
    size_t miniBatchSize;
    bool randInit;
};
struct Rewards
{
    float normalReward;
    float killedByOpponentReward;
    float outOfBoundsReward;
    float reachedGoalReward;
};
struct SimStateParams
{
    size_t traceSize;
    size_t agentVisionGridSize;
    size_t opponentVisionGridSize;
    float randomOpCoef;
};
struct OpTrackParams
{
    double pValueThreshold;
    size_t minHistorySize;
    size_t maxHistorySize;
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
    return lhs.x == rhs.x and lhs.y == rhs.y;
}

inline std::ostream &operator<<(std::ostream &out, Position pos)
{
    out << pos.x << " " << pos.y << std::endl;
    return out;
}

#endif
