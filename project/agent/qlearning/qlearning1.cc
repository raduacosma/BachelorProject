//#define XERR
#include "qlearning.ih"
// should change the constructor to take nrEpisodes and pass it to Agent
QLearning::QLearning(size_t nrEpisodes, double discountFactor, double stepSize, double epsilon)
  : Agent(nrEpisodes, epsilon),
    d_gamma(discountFactor),
    d_alpha(stepSize)
{
}
