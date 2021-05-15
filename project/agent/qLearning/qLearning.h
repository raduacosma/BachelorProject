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
    QLearning(size_t _nrEpisodes = 10000, OpModellingType pOpModellingType = OpModellingType::ONEFORALL,
              float _alpha = 0.001, float _epsilon = 0.1, float _gamma = 0.9);
    ~QLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;

  private:
};

#endif
