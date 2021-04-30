#ifndef _INCLUDED_SARSA
#define _INCLUDED_SARSA

#include "../agent.h"
#include "../../Eigen/Core"
#include "../../mlp/mlp.h"
#include "../../createRngObj/createRngObj.h"
class Sarsa : public Agent
{

    float alpha;
    float epsilon;
    float gamma;
    size_t lastAction;
    float lastQValue;
    MLP mlp;

  public:
    Sarsa(size_t _nrEpisodes, float _alpha = 0.1, float _epsilon = 0.1, float _gamma=0.9);
    ~Sarsa() override;
    bool performOneStep() override;
    std::pair<size_t,float> actionWithQ(Eigen::VectorXf const &state);
    void newEpisode() override;
  private:
};

#endif
