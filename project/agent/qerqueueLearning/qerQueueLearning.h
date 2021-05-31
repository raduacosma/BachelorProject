#ifndef _INCLUDED_QERQueueLearning
#define _INCLUDED_QERQueueLearning

#include "../../../Eigen/Core"
#include "../../createRngObj/createRngObj.h"
#include "../../mlp/mlp.h"
#include "../../utilities/utilities.h"
#include "../agent.h"
#include "../experience.h"
#include <deque>

class QERQueueLearning : public Agent
{

    float alpha;
    float epsilon;
    size_t expCounter = 0;
    bool shouldGatherExperience = true;
    size_t cCounter = 0;

    size_t lastAction;
    MLP targetMLP;
    std::deque<Experience> experiences;
    size_t cSwapPeriod;
    size_t miniBatchSize;
    size_t sizeExperience;

  public:
    QERQueueLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
                     MLPParams opponentMLP, ExpReplayParams expReplayParams, size_t _nrEpisodes,
                     OpModellingType pOpModellingType, float _alpha,
                     float _epsilon, float _gamma);
    ~QERQueueLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals) override;
    void newEpisode() override;

  private:
    void updateWithExperienceReplay();
    void handleExperience();
};

#endif
