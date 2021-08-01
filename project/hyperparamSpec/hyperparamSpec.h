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

#ifndef _INCLUDED_HYPERPARAMSPEC
#define _INCLUDED_HYPERPARAMSPEC
#include "../agent/agent.h"
#include <cstddef>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>

struct HyperparamSpec
{
    unsigned int seed;
    std::string files;
    size_t miniBatchSize;
    size_t numberOfEpisodes;
    size_t sizeExperience;
    AgentType agentType;
    float epsilon;
    float gamma;
    size_t agentVisionGridSize;
    size_t opponentVisionGridSize;
    size_t swapPeriod;
    size_t maxNrSteps;
    size_t nrRollouts;
    float agentLearningRate;
    float agentRegParam;
    float opponentLearningRate;
    float opponentRegParam;
    size_t traceSize;
    float randomOpCoef;
    OpModellingType opModellingType;
    float pValueThreshold;
    size_t minHistorySize;
    size_t maxHistorySize;
};
std::ostream &operator<<(std::ostream &out, HyperparamSpec const &hs);
std::istream &operator>>(std::istream &in, HyperparamSpec &hs);

template <typename Field>
void writeField(std::istream &in, std::string const &fieldName, Field &field)
{
    std::string label;
    in >> label;
    if (label != fieldName)
        throw std::runtime_error("Wrong parse for: " + fieldName);
    in >> field;
}

#endif
