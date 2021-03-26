#include "runHeadless.h"

#include "simContainer/simContainer.h"
#include "agent/qlearning/qlearning.h"
#include <memory>

void runHeadless(std::string const &fileList, unsigned long nrEpisodes)
{
    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent = std::make_unique<QLearning>(100,0.1,0.1,0.1);
    SimContainer simContainer{ fileList, agent.get() };
    while(simContainer.getEpisodeCount() < nrEpisodes)
        agent->performOneStep();
}
