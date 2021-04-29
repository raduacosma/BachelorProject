#ifndef _INCLUDED_SARSA
#define _INCLUDED_SARSA

#include "../agent.h"
#include "../../Eigen/Core"
#include "../../mlp/mlp.h"
class Sarsa : public Agent
{

    float alpha;
    float epsilon;

  public:
    Sarsa(size_t _nrEpisodes, float _alpha = 0.1, float _epsilon = 0.1);
    ~Sarsa() override = default;

    Actions action(Eigen::VectorXf const &state) override;
    void giveFeedback(float reward, Eigen::VectorXf const &newState) override;
  private:
};

#endif
