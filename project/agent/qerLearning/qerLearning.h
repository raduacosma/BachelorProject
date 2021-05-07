#ifndef _INCLUDED_QERLearning
#define _INCLUDED_QERLearning

#include "../agent.h"
#include "../../Eigen/Core"
#include "../../mlp/mlp.h"
#include "../../createRngObj/createRngObj.h"

struct Experience
{
    size_t action;
    float reward;
    bool isTerminal;
    Eigen::VectorXf lastState;
    Eigen::VectorXf newState;
};

class QERLearning : public Agent
{

    float alpha;
    float epsilon;
    float gamma;
    size_t C;
    int expCounter = -1;
    size_t expResetPeriod = 100000;
    bool shouldGatherExperience = true;
    int cCounter = -1;
    size_t cSwapPeriod = 1000;
    size_t miniBatchSize = 16;
    size_t sizeExperience = 10000;
    size_t lastAction;
    MLP mlp;
    MLP targetMLP;
    std::vector<Experience> experiences;

  public:
    QERLearning(size_t _nrEpisodes = 10000, float _alpha = 0.001, float _epsilon = 0.1, float _gamma=0.9);
    ~QERLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;
  private:
    void updateWithExperienceReplay();
};


#endif
