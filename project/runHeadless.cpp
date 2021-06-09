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
#include "createRngObj/createRngObj.h"

RandObj globalRng;
void runHeadless(std::string const &fileList, unsigned long nrEpisodes)
{

    //    std::cout.setstate(std::ios_base::failbit);
    std::string files = "longClock.txt,longCounter.txt,complex.txt";
    size_t cMiniBatchSize = 16;
    size_t numberOfEpisodes = 10000;   // ignore the function parameter for now until proper framework is in place
    size_t nrEpisodesToEpsilonZero = numberOfEpisodes/4*3;
    size_t sizeExperience = 10000;
    float alpha = 0.001;
    float epsilon = 0.5;
    float gamma = 0.9;
    globalRng = RandObj(275165314,-1.0f,1.0f,sizeExperience);
    OpModellingType opModellingType=OpModellingType::ONEFORALL;
    ExpReplayParams expReplayParams{ .cSwapPeriod = 1000, .miniBatchSize = cMiniBatchSize, .sizeExperience = sizeExperience };
    AgentMonteCarloParams agentMonteCarloParams{ .maxNrSteps = 1, .nrRollouts = 5 };
    MLPParams agentMLP{ .sizes = { 52, 200, 4 },
                        .learningRate = 0.001,
                        .outputActivationFunc = ActivationFunction::LINEAR,
                        .miniBatchSize = cMiniBatchSize };
    MLPParams opponentMLP{ .sizes = { 75, 200, 4 },
                           .learningRate = 0.001,
                           .outputActivationFunc = ActivationFunction::SOFTMAX,
                           .miniBatchSize = cMiniBatchSize };
    Rewards rewards = {
        .normalReward = -0.1f, .killedByOpponentReward = -100.0f, .outOfBoundsReward = -0.1f, .reachedGoalReward = 100.0f
    };
    SimStateParams simStateParams = { .traceSize = 6, .visionGridSize = 2, .randomOpCoef=-1 };
    OpTrackParams kolsmirParams = { .pValueThreshold = 0.05, .minHistorySize = 10, .maxHistorySize = 10 };
    OpTrackParams pettittParams = { .pValueThreshold = 0.01, .minHistorySize = 10, .maxHistorySize = 20 };

    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent =
        std::make_unique<QERQueueLearning>(kolsmirParams, agentMonteCarloParams, agentMLP, opponentMLP, expReplayParams,
                                           numberOfEpisodes,nrEpisodesToEpsilonZero, OpModellingType::ONEFORALL,alpha,epsilon,gamma);
    SimContainer simContainer{ files, agent.get(), rewards, simStateParams };
    agent->run();
    std::ofstream out{ "results/rewards04LESS.txt" };
    std::vector<float> const &agentRewards = agent->getRewards();
    copy(agentRewards.begin(), agentRewards.end(), std::ostream_iterator<float>(out, "\n"));
    std::ofstream opponent{ "results/opponentPredictionLossesTwoDOUBLE.txt" };
    std::vector<float> const &opponentPred = agent->getOpponentPredictionLosses();
    copy(opponentPred.begin(), opponentPred.end(), std::ostream_iterator<float>(opponent, "\n"));
    std::ofstream opponentPerc{ "results/opponentPredictionPercentageTwoDOUBLE.txt" };
    std::vector<float> const &opponentPredPerc = agent->getOpponentCorrectPredictionPercentage();
    copy(opponentPredPerc.begin(), opponentPredPerc.end(), std::ostream_iterator<float>(opponentPerc, "\n"));

    std::cout << "opponent prediction percentage: " << agent->getCorrectOpponentTypePredictionPercentage() << std::endl;
    std::cout << "nr of times killed by opponent: "<< agent->getOpDeathPercentage()<<std::endl;
    //    std::ofstream opponentLoss{"results/opponentFirstEpLoss.txt"};
    //    std::vector<float> const &opponentThisLoss = agent->getThisEpisodeLoss();
    //    copy(opponentThisLoss.begin(), opponentThisLoss.end(),
    //         std::ostream_iterator<float>(opponentLoss, "\n"));
}
