#include "randobj.h"

RandObj::RandObj(float minMLP, float maxMLP, int expReplaySize)
    : rngEngine(std::random_device{}()), uniReal01(0.0, 1.0), randomInitMLP(minMLP, maxMLP),
      expReplayIdx(0, expReplaySize - 1)
{
}
RandObj::RandObj(unsigned int seed, float minMLP, float maxMLP, int expReplaySize)
    : rngEngine(seed), uniReal01(0.0, 1.0), randomInitMLP(minMLP, maxMLP), expReplayIdx(0, expReplaySize - 1)
{
}
