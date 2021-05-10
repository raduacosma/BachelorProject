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
    std::uniform_int_distribution<int> expReplayIdx;

  public:
    RandObj();
    explicit RandObj(unsigned int seed);
    float getUniReal01();
    int getExpReplayIdx();
};
#endif
