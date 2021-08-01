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

#ifndef _INCLUDED_MONTECARLOSIM
#define _INCLUDED_MONTECARLOSIM

#include "../../Eigen/Core"
#include "../utilities/utilities.h"
#include <deque>
#include <string>
#include <vector>

class Agent;

#include "../agent/agent.h"
#include "../simState/simState.h"

class MonteCarloSim
{

    Position agentPos;
    Position simSize;
    Position goalPos;
    size_t traceSize;
    size_t agentVisionGridSize;
    size_t agentVisionGridSideSize; // visionGridSize*2+1
    size_t agentStateSize;          // visionGridSizeSize^2
    size_t opponentVisionGridSize;
    size_t opponentVisionGridSideSize; // visionGridSize*2+1
    size_t opponentStateSize;          // visionGridSizeSize^2

    std::vector<Position> const &walls;

    // REWARDS
    float d_outOfBoundsReward;
    float d_reachedGoalReward;
    float d_killedByOpponentReward;
    float d_normalReward;
    std::deque<Position> opponentTrace;

  public:
    MonteCarloSim(SimState const &simState);
    std::tuple<float, SimResult> computeNextStateAndReward(Actions action, Actions opAction);
    Eigen::VectorXf getStateForAgent() const;
    Eigen::VectorXf getStateForOpponent() const;
    void updateOpPos(Actions opAction);

  private:
    std::pair<float, SimResult> updateAgentPos(Actions action);
    Position computeNewPos(Actions currAction, Position pos);
};

#endif
