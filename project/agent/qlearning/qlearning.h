#ifndef INCLUDED_QLEARNING_
#define INCLUDED_QLEARNING_

#include "../agent.h"
#include <vector>

class QLearning : public Agent
{
  float d_gamma;
  float d_alpha;
  std::vector<std::vector<float>> d_QTable;

  public:
    QLearning(size_t nrEpisodes, float discountFactor, float stepSize = 0.1, float epsilon = 0.1);
    ~QLearning();

    virtual Actions action(size_t stateIdx) override;
    virtual void giveFeedback(float reward, size_t newState) override;

    virtual void stateSpaceSize(size_t size) override;
  private:
};

inline QLearning::~QLearning()
{

}
        
#endif
