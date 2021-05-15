#ifndef _INCLUDED_DQERQueueLearning
#define _INCLUDED_DQERQueueLearning

#include "../../../Eigen/Core"
#include "../../createRngObj/createRngObj.h"
#include "../../mlp/mlp.h"
#include "../agent.h"
#include "../experience.h"
#include <deque>

class DQERQueueLearning : public Agent
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
    DQERQueueLearning(size_t _nrEpisodes = 10000, OpModellingType pOpModellingType = OpModellingType::ONEFORALL,
                      float _alpha = 0.001, float _epsilon = 0.1, float _gamma = 0.9);
    ~DQERQueueLearning() override;
    bool performOneStep() override;
    size_t actionWithQ(Eigen::VectorXf const &qVals);
    void newEpisode() override;

  private:
    void updateWithExperienceReplay();
    void handleExperience();
};

#endif
