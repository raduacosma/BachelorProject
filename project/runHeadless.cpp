#include "runHeadless.h"

#include "simContainer/simContainer.h"
#include "agent/sarsa/sarsa.h"
#include "agent/qLearning/qLearning.h"
#include <memory>
#include <fstream>
#include <iterator>
#include <algorithm>

void runHeadless(std::string const &fileList, unsigned long nrEpisodes)
{
    std::string files = "simple_wall.txt";
    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent = std::make_unique<QLearning>(10000);
    SimContainer simContainer{ files, agent.get() };
    agent->run();
    std::ofstream out{"results/rewardsQWall.txt"};
    std::vector<float> const &rewards = agent->getRewards();
    copy(rewards.begin(), rewards.end(),
         std::ostream_iterator<float>(out, "\n"));
    std::ofstream opponent{"results/opponentPredictionLossesQ.txt"};
    std::vector<float> const &opponentPred = agent->getOpponentPredictionLosses();
    copy(opponentPred.begin(), opponentPred.end(),
         std::ostream_iterator<float>(opponent, "\n"));
    std::ofstream opponentPerc{"results/opponentPredictionPercentageQ.txt"};
    std::vector<float> const &opponentPredPerc = agent->getOpponentCorrectPredictionPercentage();
    copy(opponentPredPerc.begin(), opponentPredPerc.end(),
         std::ostream_iterator<float>(opponentPerc, "\n"));
}
