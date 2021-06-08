#ifndef _INCLUDED_QERLearning
#define _INCLUDED_QERLearning

#include "../../Eigen/Core"
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
                OpModellingType pOpModellingType, float pAlpha,
                float pEpsilon, float pGamma);
    ~QERLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;

  private:
    void updateWithExperienceReplay();
};

#endif
