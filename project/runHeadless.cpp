#include "runHeadless.h"

#include "simContainer/simContainer.h"
#include "agent/qlearning/qlearning.h"

void runHeadless(std::string const &fileList, unsigned long nrEpisodes)
{
    Agent * agent = new QLearning(100,0.1,0.1,0.1);
    SimContainer simContainer{ fileList, agent };
    while(simContainer.getEpisodeCount() < nrEpisodes)
        agent->performOneStep();
    delete agent;
}
