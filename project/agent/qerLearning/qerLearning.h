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

#ifndef _INCLUDED_QERLearning
#define _INCLUDED_QERLearning

#include "../../../Eigen/Core"
#include "../../createRngObj/createRngObj.h"
#include "../../mlp/mlp.h"
#include "../agent.h"
#include "../experience.h"

class QERLearning : public Agent
{
    size_t C;
    size_t expCounter = 0;
    size_t expResetPeriod = 100000;
    bool shouldGatherExperience = true;
    size_t cCounter = 0;

    size_t lastAction;
    MLP targetMLP;
    size_t cSwapPeriod;
    size_t miniBatchSize;
    size_t sizeExperience;
    std::vector<Experience> experiences;

  public:
    QERLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
                MLPParams opponentMLP, ExpReplayParams expReplayParams, size_t _nrEpisodes,
                size_t pNrEpisodesToEpsilonZero, OpModellingType pOpModellingType, float pEpsilon, float pGamma);
    ~QERLearning() override;
    bool performOneStep() override;
    void newEpisode() override;

  private:
    void updateWithExperienceReplay();
};

#endif
