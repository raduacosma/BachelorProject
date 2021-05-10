#ifndef _INCLUDED_QERQueueLearning
#define _INCLUDED_QERQueueLearning

#include "../agent.h"
#include "../../Eigen/Core"
#include "../../mlp/mlp.h"
#include "../../createRngObj/createRngObj.h"
#include "../experience.h"
#include <deque>

class QERQueueLearning : public Agent
{

    float alpha;
    float epsilon;
    size_t C;
    size_t expCounter = 0;
    bool shouldGatherExperience = true;
    size_t cCounter = 0;
    size_t cSwapPeriod = 1000;
    size_t miniBatchSize = 16;
    size_t sizeExperience = 10000;
    size_t lastAction;
    MLP targetMLP;
    std::deque<Experience> experiences;

  public:
    QERQueueLearning(size_t _nrEpisodes = 10000, OpModellingType pOpModellingType = OpModellingType::ONEFORALL, float _alpha = 0.001, float _epsilon = 0.1, float _gamma=0.9);
    ~QERQueueLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;
  private:
    void updateWithExperienceReplay();
    void handleExperience();
};


#endif
