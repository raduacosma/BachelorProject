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

#ifndef _INCLUDED_OPTRACK
#define _INCLUDED_OPTRACK

class Agent;
#include "../agent/experience.h"
#include "../mlp/mlp.h"
#include <deque>
#include <vector>

class OpTrack
{
    double pValueThreshold = 0.01;
    std::vector<std::vector<OpExperience>> opListStateHistory;
    std::vector<std::deque<OpExperience>> opDequeStateHistory;
    std::vector<std::vector<double>> opListLossHistory;
    std::vector<std::deque<double>> opDequeLossHistory;
    std::vector<OpExperience> currOpListStateHistory;
    std::vector<double> currOpListLossHistory;

    std::vector<MLP> opCopies;
    bool firstTime = true;
    bool foundOpModel = false;

  public:
    bool isFoundOpModel() const;

  private:
    size_t opHistoryCounter = 0;
    size_t minHistorySize = 10;
    size_t maxHistorySize = 20;

  public:
    OpTrack(double pPValueThreshold, size_t pMinHistorySize, size_t pMaxHistorySize);
    void normalOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState, float loss);
    void kolsmirOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState, float loss);
    void pettittOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState, float loss);
    void noTrainPettittOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                                  float loss);
    void kolsmirOpInit(Agent &agent);
    void noTrainPettittOpInit(Agent &agent);
    void pettittOpInit(Agent &agent);
    void commonOpInit(Agent &agent);

    void destroyRandomPettitt(Agent &agent);
    void destroyRandomNoTrainPettitt(Agent &agent);
    void destroyRandomKolsmir(Agent &agent);
    static void updateCorrectPercentage(Agent &agent);
};

inline bool OpTrack::isFoundOpModel() const
{
    return foundOpModel;
}

#endif
