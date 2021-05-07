#include "randobj.h"

RandObj::RandObj() : rngEngine(std::random_device{}()), uniReal01(0.0, 1.0), expReplayIdx(0,9999)
{
}
RandObj::RandObj(unsigned int seed) : rngEngine(seed), uniReal01(0.0, 1.0), expReplayIdx(0,9999)
{
}
float RandObj::getUniReal01()
{
    return uniReal01(rngEngine);
}
int RandObj::getExpReplayIdx()
{
   return expReplayIdx(rngEngine);
}
