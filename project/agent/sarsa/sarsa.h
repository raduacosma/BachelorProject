#ifndef _INCLUDED_SARSA
#define _INCLUDED_SARSA

#include "../../Eigen/Core"
#include "../../createRngObj/createRngObj.h"
#include "../../mlp/mlp.h"
#include "../agent.h"
class Sarsa : public Agent
{
    size_t lastAction;
    Eigen::VectorXf lastQValues;

  public:
    Sarsa(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
          MLPParams opponentMLP, size_t _nrEpisodes,
          OpModellingType pOpModellingType, float pAlpha, float pEpsilon,
          float pGamma);
    ~Sarsa() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;

  private:
};

#endif
