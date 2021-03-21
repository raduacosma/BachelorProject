#ifndef INCLUDED_QLEARNING_
#define INCLUDED_QLEARNING_

#include "../agent.h"
#include <vector>

class QLearning : public Agent
{
  double d_gamma;
  double d_alpha;
  std::vector<std::vector<double>> d_QTable;

  public:
    QLearning(size_t nrEpisodes, double discountFactor, double stepSize = 0.1, double epsilon = 0.1);
    ~QLearning();

    virtual Actions action(size_t stateIdx) override;
    virtual void giveFeedback(double reward, size_t newState) override;

    virtual void stateSpaceSize(size_t size) override;
  private:
};

inline QLearning::~QLearning()
{

}
        
#endif
