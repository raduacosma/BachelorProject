/*
     Copyright (C) 2021  Radu Alexandru Cosma

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

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
