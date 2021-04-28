#ifndef _INCLUDED_RANDOBJ
#define _INCLUDED_RANDOBJ

#include <random>

class RandObj
{
    std::mt19937 rngEngine;
    std::uniform_real_distribution<float> uniReal01;

  public:
    RandObj();
    explicit RandObj(unsigned int seed);
    float getUniReal01();
};
#endif
