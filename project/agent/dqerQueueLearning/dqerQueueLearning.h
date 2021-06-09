#ifndef _INCLUDED_DQERQueueLearning
#define _INCLUDED_DQERQueueLearning

#include "../../../Eigen/Core"
#include "../../createRngObj/createRngObj.h"
#include "../../mlp/mlp.h"
#include "../../utilities/utilities.h"
#include "../agent.h"
#include "../experience.h"
#include <deque>

class DQERQueueLearning : public Agent
{
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
    DQERQueueLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
                      MLPParams opponentMLP, ExpReplayParams expReplayParams, size_t _nrEpisodes,
                      size_t pNrEpisodesToEpsilonZero, OpModellingType pOpModellingType, float pAlpha, float pEpsilon,
                      float pGamma);
    ~DQERQueueLearning() override;
    bool performOneStep() override;
    void newEpisode() override;

  private:
    void updateWithExperienceReplay();
    void handleExperience();
};

#endif
