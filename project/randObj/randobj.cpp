#include "randobj.h"
// TODO: make the limits for distrs configurable
RandObj::RandObj() : rngEngine(std::random_device{}()), uniReal01(0.0, 1.0), uniReal44(-1.0f,1.0f),expReplayIdx(0, 9999)
{
}
RandObj::RandObj(unsigned int seed) : rngEngine(seed), uniReal01(0.0, 1.0),uniReal44(-1.0f,1.0f), expReplayIdx(0, 9999)
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
std::mt19937 &RandObj::getRngEngine()
{
    return rngEngine;
}
float RandObj::getUniReal44()
{
    return uniReal44(rngEngine);
}
