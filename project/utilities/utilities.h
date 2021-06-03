#ifndef _INCLUDED_UTILITIES
#define _INCLUDED_UTILITIES

#include <cstddef>
#include <ostream>
#include <vector>
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
    ActivationFunction outputActivationFunc;
    size_t miniBatchSize;
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
    size_t visionGridSize;
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
