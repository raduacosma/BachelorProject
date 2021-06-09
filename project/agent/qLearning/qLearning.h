#ifndef _INCLUDED_QLEARNING
#define _INCLUDED_QLEARNING

#include "../../../Eigen/Core"
#include "../../createRngObj/createRngObj.h"
#include "../../mlp/mlp.h"
#include "../agent.h"
class QLearning : public Agent
{

    size_t lastAction;

  public:
    QLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
              MLPParams opponentMLP, size_t _nrEpisodes,size_t pNrEpisodesToEpsilonZero,
              OpModellingType pOpModellingType, float pAlpha, float pEpsilon,
              float pGamma);
    ~QLearning() override;
    bool performOneStep() override;
    void newEpisode() override;

  private:
};

#endif
