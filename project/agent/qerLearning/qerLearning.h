#ifndef _INCLUDED_QERLearning
#define _INCLUDED_QERLearning

#include "../../Eigen/Core"
#include "../../createRngObj/createRngObj.h"
#include "../../mlp/mlp.h"
#include "../agent.h"
#include "../experience.h"

class QERLearning : public Agent
{

    float alpha;
    float epsilon;
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
                MLPParams opponentMLP, ExpReplayParams expReplayParams, size_t _nrEpisodes = 10000,
                OpModellingType pOpModellingType = OpModellingType::ONEFORALL, float _alpha = 0.001,
                float _epsilon = 0.1, float _gamma = 0.9);
    ~QERLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;

  private:
    void updateWithExperienceReplay();
};

#endif
