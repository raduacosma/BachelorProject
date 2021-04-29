#include "qlearning.ih"

Actions QLearning::action(Eigen::VectorXf const &state)
{
//  bool explore = Random::uniform() < EPSILON;
//
//  if (explore)
//    d_lastAction = Random::uniform() * NR_ACTIONS;
//  else
//  {
//    double max = std::numeric_limits<double>::lowest();
//    for (size_t action = 0; action != NR_ACTIONS; ++action)
//      if (d_QTable.at(stateIdx)[action] > max)
//      {
//        max = d_QTable.at(stateIdx)[action];
//        d_lastAction = action;
//      }
//  }
lastAction = globalRng.getUniReal01() * NR_ACTIONS;
  return static_cast<Actions>(lastAction);
}
