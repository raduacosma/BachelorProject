/*
The MIT License (MIT)

Copyright (c) 2014-2021 Omar Cornut
Copyright (c) 2021 Radu Alexandru Cosma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

// The FloatVec structs were copied from ImGUI (https://github.com/ocornut/imgui)
// in order to remove the dependency on it for non-GUI purposes

#ifndef _INCLUDED_UTILITIES
#define _INCLUDED_UTILITIES

#include <cstddef>
#include <ostream>
#include <vector>
struct FloatVec2
{
    float                                   x, y;
    FloatVec2()                                { x = y = 0.0f; }
    FloatVec2(float _x, float _y)              { x = _x; y = _y; }
    float  operator[] (size_t idx) const    { return (&x)[idx]; }
    float& operator[] (size_t idx)          { return (&x)[idx]; }
#ifdef IM_VEC2_CLASS_EXTRA
    IM_VEC2_CLASS_EXTRA     // Define additional constructors and implicit cast operators in imconfig.h to convert back and forth between your math types and ImVec2.
#endif
};


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
