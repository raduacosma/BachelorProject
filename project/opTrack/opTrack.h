#ifndef _INCLUDED_OPTRACK
#define _INCLUDED_OPTRACK

class Agent;
#include <vector>
#include "../agent/experience.h"
#include "../mlp/mlp.h"
#include <deque>

class OpTrack
{
    double pValueThreshold = 0.05;
    std::vector<std::vector<OpExperience>> opListStateHistory;
    std::vector<std::vector<double>> opListLossHistory;
    std::vector<std::deque<double>> opDequeLossHistory;
    std::vector<OpExperience> currOpListStateHistory;
    std::vector<double> currOpListLossHistory;

    std::vector<MLP> opCopies;
    bool firstTime = true;
    bool foundOpModel = false;
    size_t opHistoryCounter = 0;
    size_t minHistorySize = 10;
    size_t maxHistorySize = 10;

  public:
    void normalOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState, float loss);
    void kolsmirOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState, float loss);
    void pettittOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState, float loss);
    void kolsmirOpInit(Agent &agent);
    void pettittOpInit(Agent &agent);
    void commonOpInit(Agent &agent);

    void destroyRandomPettitt(Agent &agent);
    void destroyRandomKolsmir(Agent &agent);
    void updateCorrectPercentage(Agent &agent);
};
#endif
