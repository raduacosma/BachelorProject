#include "runHeadless.h"

#include "agent/dqerQueueLearning/dqerQueueLearning.h"
#include "agent/qLearning/qLearning.h"
#include "agent/qerLearning/qerLearning.h"
#include "agent/qerqueueLearning/qerQueueLearning.h"
#include "agent/sarsa/sarsa.h"
#include "simContainer/simContainer.h"
#include "utilities/utilities.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>

void runHeadless(std::string const &fileList, unsigned long nrEpisodes)
{
    //    std::cout.setstate(std::ios_base::failbit);
    std::string files = "longClock.txt,longCounter.txt,complex.txt";
    size_t cMiniBatchSize = 16;
    size_t numberOfEpisodes = 10000;   // ignore the parameter for now until proper framework is in place
    float alpha = 0.001;
    float epsilon = 0.1;
    float gamma = 0.9;
    OpModellingType opModellingType=OpModellingType::ONEFORALL;
    ExpReplayParams expReplayParams{ .cSwapPeriod = 1000, .miniBatchSize = cMiniBatchSize, .sizeExperience = 10000 };
    AgentMonteCarloParams agentMonteCarloParams{ .maxNrSteps = 1, .nrRollouts = 5 };
    MLPParams agentMLP{ .sizes = { 52, 192, 4 },
                        .learningRate = 0.001,
                        .outputActivationFunc = ActivationFunction::LINEAR,
                        .miniBatchSize = cMiniBatchSize };
    MLPParams opponentMLP{ .sizes = { 50, 100, 4 },
                           .learningRate = 0.001,
                           .outputActivationFunc = ActivationFunction::SOFTMAX,
                           .miniBatchSize = cMiniBatchSize };
    Rewards rewards = {
        .normalReward = -0.1, .killedByOpponentReward = -100, .outOfBoundsReward = -0.1, .reachedGoalReward = 100
    };
    SimStateParams simStateParams = { .traceSize = 6, .visionGridSize = 2, .randomOpCoef=-1 };
    OpTrackParams kolsmirParams = { .pValueThreshold = 0.05, .minHistorySize = 10, .maxHistorySize = 10 };
    OpTrackParams pettittParams = { .pValueThreshold = 0.01, .minHistorySize = 10, .maxHistorySize = 20 };

    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent =
        std::make_unique<QERQueueLearning>(pettittParams, agentMonteCarloParams, agentMLP, opponentMLP, expReplayParams,
                                           numberOfEpisodes, OpModellingType::NOTRAINPETTITT,alpha,epsilon,gamma);
    SimContainer simContainer{ files, agent.get(), rewards, simStateParams };
    agent->run();
    std::ofstream out{ "results/rewardsDQER.txt" };
    std::vector<float> const &agentRewards = agent->getRewards();
    copy(agentRewards.begin(), agentRewards.end(), std::ostream_iterator<float>(out, "\n"));
    std::ofstream opponent{ "results/opponentPredictionLossesTwo.txt" };
    std::vector<float> const &opponentPred = agent->getOpponentPredictionLosses();
    copy(opponentPred.begin(), opponentPred.end(), std::ostream_iterator<float>(opponent, "\n"));
    std::ofstream opponentPerc{ "results/opponentPredictionPercentageTwo.txt" };
    std::vector<float> const &opponentPredPerc = agent->getOpponentCorrectPredictionPercentage();
    copy(opponentPredPerc.begin(), opponentPredPerc.end(), std::ostream_iterator<float>(opponentPerc, "\n"));

    std::cout << "opponent prediction percentage: " << agent->getCorrectOpponentTypePredictionPercentage() << std::endl;

    //    std::ofstream opponentLoss{"results/opponentFirstEpLoss.txt"};
    //    std::vector<float> const &opponentThisLoss = agent->getThisEpisodeLoss();
    //    copy(opponentThisLoss.begin(), opponentThisLoss.end(),
    //         std::ostream_iterator<float>(opponentLoss, "\n"));
}
