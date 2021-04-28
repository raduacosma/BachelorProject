#include "agent.h"
#include "tuple"
#include <iostream>
#include "../simContainer/simContainer.h"
using namespace std;
bool Agent::performOneStep()
{
    auto [reward, newState, canContinue] =
    d_maze->computeNextStateAndReward(action(d_oldstate));
    giveFeedback(reward, newState);
//    totalReward += reward;
    if (not canContinue)
    {
        return false;
    }
    // check if d_oldstate should be updated even if we can't continue
    d_oldstate = newState;
    return true;
}
void Agent::run()
{
    // initialize Q values? They are set to 0 in stateSpaceSize and Marco's
    // slides say initialize "arbitrarily" while the book says terminal states
    // should be 0, so I guess initializing them all to 0 could be fine?

    // tracking stuff? avg rewards etc etc etc
    d_runReward = 0;
    for (size_t nrEpisode = 0; nrEpisode != d_nrEpisodes; ++nrEpisode)
    {
        newEpisode(d_maze->mazeStateHash()); // d_oldstate was modified from Maze
        // so it's fine
        // anything else?
        float totalReward = 0;
        while (true)
        {
            if (not performOneStep())
                break;
        }
        d_runReward += totalReward;
        d_rewards[nrEpisode] = totalReward;
    }
    // any cleanup?
}

Agent::Agent(size_t nrEpisodes, float epsilon)
    : d_nrEpisodes(nrEpisodes), d_rewards(vector<float>(nrEpisodes)),
      d_hasDied(vector<size_t>(nrEpisodes)), EPSILON(epsilon)
{
}


void Agent::maze(SimContainer *maze)
{
    d_maze = maze;
}
void Agent::newEpisode(size_t stateIdx)
{
    d_oldstate = stateIdx;
    // Some algorithms require this. Empty for the others.
}

vector<float> &Agent::rewards()
{
    return d_rewards;
}

void Agent::initialState(size_t state)
{
    d_oldstate = state;
}

vector<size_t> & Agent::hasDied()
{
    return d_hasDied;
}

float Agent::runReward()
{
    return d_runReward;
}

