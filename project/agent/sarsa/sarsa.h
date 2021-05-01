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
    Eigen::VectorXf lastOpponentState;
    Eigen::VectorXf lastQValues;
    MLP mlp;
    MLP opponentMlp;

  public:
    Sarsa(size_t _nrEpisodes = 10000, float _alpha = 0.001, float _epsilon = 0.1, float _gamma=0.9);
    ~Sarsa() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;
  private:
};

#endif
