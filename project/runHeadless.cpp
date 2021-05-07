#include "runHeadless.h"

#include "simContainer/simContainer.h"
#include "agent/sarsa/sarsa.h"
#include "agent/qLearning/qLearning.h"
#include "agent/qerLearning/qerLearning.h"
#include <memory>
#include <fstream>
#include <iterator>
#include <algorithm>

void runHeadless(std::string const &fileList, unsigned long nrEpisodes)
{
    std::string files = "simple_opponent.txt";
    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent = std::make_unique<QERLearning>(10000);
    SimContainer simContainer{ files, agent.get() };
    agent->run();
    std::ofstream out{"results/rewardsTwoSimple.txt"};
    std::vector<float> const &rewards = agent->getRewards();
    copy(rewards.begin(), rewards.end(),
         std::ostream_iterator<float>(out, "\n"));
    std::ofstream opponent{"results/opponentPredictionLossesTwo.txt"};
    std::vector<float> const &opponentPred = agent->getOpponentPredictionLosses();
    copy(opponentPred.begin(), opponentPred.end(),
         std::ostream_iterator<float>(opponent, "\n"));
    std::ofstream opponentPerc{"results/opponentPredictionPercentageTwo.txt"};
    std::vector<float> const &opponentPredPerc = agent->getOpponentCorrectPredictionPercentage();
    copy(opponentPredPerc.begin(), opponentPredPerc.end(),
         std::ostream_iterator<float>(opponentPerc, "\n"));

//    std::ofstream opponentLoss{"results/opponentFirstEpLoss.txt"};
//    std::vector<float> const &opponentThisLoss = agent->getThisEpisodeLoss();
//    copy(opponentThisLoss.begin(), opponentThisLoss.end(),
//         std::ostream_iterator<float>(opponentLoss, "\n"));
}
