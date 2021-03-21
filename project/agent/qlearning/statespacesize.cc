#include "qlearning.ih"

// Set the size of the state space
void QLearning::stateSpaceSize(size_t size)
{
  d_QTable.clear();

  vector<double> initialValues { Q_0, Q_0, Q_0, Q_0 };
  for (size_t idx = 0; idx != size; ++idx)
    d_QTable.push_back(initialValues);
}