#ifndef _INCLUDED_QLEARNING
#define _INCLUDED_QLEARNING

#include "../agent.h"
#include "../../Eigen/Core"
#include "../../mlp/mlp.h"
#include "../../createRngObj/createRngObj.h"
class QLearning : public Agent
{

    float alpha;
    float epsilon;
    float gamma;
    size_t lastAction;

  public:
    QLearning(size_t _nrEpisodes = 10000, float _alpha = 0.001, float _epsilon = 0.1, float _gamma=0.9);
    ~QLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;
  private:
};

#endif
