#ifndef _INCLUDED_RANDOBJ
#define _INCLUDED_RANDOBJ

#include <random>

class RandObj
{
    std::mt19937 rngEngine;

  public:
    std::mt19937 &getRngEngine();

  private:
    std::uniform_real_distribution<float> uniReal01;
    std::uniform_real_distribution<float> randomInitMLP;
    std::uniform_int_distribution<int> expReplayIdx;

  public:
    RandObj() = default;
    RandObj(float minMLP, float maxMLP, int expReplaySize);
    explicit RandObj(unsigned int seed, float minMLP, float maxMLP, int expReplaySize);
    float getUniReal01();
    float getRandomInitMLP();
    int getExpReplayIdx();
};
inline float RandObj::getUniReal01()
{
    return uniReal01(rngEngine);
}
inline int RandObj::getExpReplayIdx()
{
    return expReplayIdx(rngEngine);
}
inline std::mt19937 &RandObj::getRngEngine()
{
    return rngEngine;
}
inline float RandObj::getRandomInitMLP()
{
    return randomInitMLP(rngEngine);
}
#endif
