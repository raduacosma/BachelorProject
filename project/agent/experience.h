#ifndef _INCLUDED_EXPERIENCE
#define _INCLUDED_EXPERIENCE

#include "../Eigen/Core"
struct Experience
{
    size_t action;
    float reward;
    bool isTerminal;
    Eigen::VectorXf lastState;
    Eigen::VectorXf newState;
};

#endif
