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
    std::uniform_real_distribution<float> uniReal44;
    std::uniform_int_distribution<int> expReplayIdx;

  public:
    RandObj();
    explicit RandObj(unsigned int seed);
    float getUniReal01();
    float getUniReal44();
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
inline float RandObj::getUniReal44()
{
    return uniReal44(rngEngine);
}
#endif
