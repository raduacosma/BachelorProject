#include "runHeadless.h"

#include "simContainer/simContainer.h"
#include "agent/sarsa/sarsa.h"
#include <memory>
#include <fstream>
#include <iterator>
#include <algorithm>

void runHeadless(std::string const &fileList, unsigned long nrEpisodes)
{
    std::string files = "simulation_state.txt";
    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent = std::make_unique<Sarsa>(100000);
    SimContainer simContainer{ files, agent.get() };
    agent->run();
    std::ofstream out{"results/rewards.txt"};
    std::vector<float> const &rewards = agent->getRewards();
    copy(rewards.begin(), rewards.end(),
         std::ostream_iterator<float>(out, "\n"));
}
