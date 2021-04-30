#include "runHeadless.h"

#include "simContainer/simContainer.h"
#include "agent/sarsa/sarsa.h"
#include <memory>

void runHeadless(std::string const &fileList, unsigned long nrEpisodes)
{
    std::string files = "simulation_state.txt";
    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent = std::make_unique<Sarsa>(10000,0.1,0.1,0.1);
    SimContainer simContainer{ files, agent.get() };
    agent->run();
}
