#ifndef _INCLUDED_QERLearning
#define _INCLUDED_QERLearning

#include "../agent.h"
#include "../../Eigen/Core"
#include "../../mlp/mlp.h"
#include "../../createRngObj/createRngObj.h"
#include "../experience.h"


class QERLearning : public Agent
{

    float alpha;
    float epsilon;
    float gamma;
    size_t C;
    size_t expCounter = 0;
    size_t expResetPeriod = 100000;
    bool shouldGatherExperience = true;
    size_t cCounter = 0;
    size_t cSwapPeriod = 1000;
    size_t miniBatchSize = 16;
    size_t sizeExperience = 10000;
    size_t lastAction;
    MLP targetMLP;
    std::vector<Experience> experiences;

  public:
    QERLearning(size_t _nrEpisodes = 10000, OpModellingType pOpModellingType = OpModellingType::ONEFORALL, float _alpha = 0.001, float _epsilon = 0.1, float _gamma=0.9);
    ~QERLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;
  private:
    void updateWithExperienceReplay();
};


#endif
