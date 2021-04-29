#include "agent.h"
#include "../simContainer/simContainer.h"
#include "tuple"
#include <iostream>
using namespace std;
bool Agent::performOneStep()
{
    auto [reward, canContinue] = maze->computeNextStateAndReward(action(oldState));
    Eigen::VectorXf newState = maze->getStateForAgent();
    giveFeedback(reward, newState);
    if (not canContinue)
    {
        return false;
    }
    // check if d_oldstate should be updated even if we can't continue
    oldState = newState;
    return true;
}
void Agent::run()
{
    // initialize Q values? They are set to 0 in stateSpaceSize and Marco's
    // slides say initialize "arbitrarily" while the book says terminal states
    // should be 0, so I guess initializing them all to 0 could be fine?

    // tracking stuff? avg rewards etc etc etc
    runReward = 0;
    for (size_t nrEpisode = 0; nrEpisode != nrEpisodes; ++nrEpisode)
    {
        // d_oldstate was modified from Maze so it's fine, anything else?
        newEpisode();
        float totalReward = 0;
        while (true)
        {
            bool canContinue = performOneStep();
            totalReward += maze->getLastReward();
            if (not canContinue)
                break;
        }
        runReward += totalReward;
        rewards[nrEpisode] = totalReward;
    }
    // any cleanup?
}

Agent::Agent(size_t _nrEpisodes)
    : nrEpisodes(_nrEpisodes), rewards(vector<float>(_nrEpisodes)), hasDied(vector<size_t>(_nrEpisodes))
{
}

void Agent::setMaze(SimContainer *simCont)
{
    maze = simCont;
}
void Agent::newEpisode()
{
    oldState = maze->getStateForAgent();
    // Some algorithms require this. Empty for the others.
}

vector<float> &Agent::getRewards()
{
    return rewards;
}

vector<size_t> &Agent::getHasDied()
{
    return hasDied;
}

float Agent::getRunReward()
{
    return runReward;
}
