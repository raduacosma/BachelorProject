#ifndef _INCLUDED_QLEARNING
#define _INCLUDED_QLEARNING

#include "../../../Eigen/Core"
#include "../../createRngObj/createRngObj.h"
#include "../../mlp/mlp.h"
#include "../agent.h"
class QLearning : public Agent
{

    float alpha;
    float epsilon;
    size_t lastAction;

  public:
    QLearning(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
              MLPParams opponentMLP, size_t _nrEpisodes,
              OpModellingType pOpModellingType, float _alpha, float _epsilon,
              float _gamma);
    ~QLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;

  private:
};

#endif
