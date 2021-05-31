#ifndef _INCLUDED_SARSA
#define _INCLUDED_SARSA

#include "../../Eigen/Core"
#include "../../createRngObj/createRngObj.h"
#include "../../mlp/mlp.h"
#include "../agent.h"
class Sarsa : public Agent
{

    float alpha;
    float epsilon;
    size_t lastAction;
    Eigen::VectorXf lastQValues;

  public:
    Sarsa(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
          MLPParams opponentMLP, size_t _nrEpisodes = 10000,
          OpModellingType pOpModellingType = OpModellingType::ONEFORALL, float _alpha = 0.001, float _epsilon = 0.1,
          float _gamma = 0.9);
    ~Sarsa() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;

  private:
};

#endif
