#include "runHeadless.h"

#include "agent/dqerQueueLearning/dqerQueueLearning.h"
#include "agent/qLearning/qLearning.h"
#include "agent/qerLearning/qerLearning.h"
#include "agent/qerqueueLearning/qerQueueLearning.h"
#include "agent/sarsa/sarsa.h"
#include "createRngObj/createRngObj.h"
#include "simContainer/simContainer.h"
#include "utilities/utilities.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <chrono>

RandObj globalRng;
void runHeadless(std::string const &fileList, unsigned long nrEpisodes)
{
    auto begin = std::chrono::high_resolution_clock::now();
    //    std::cout.setstate(std::ios_base::failbit);
    std::string files = "try1.txt,try2.txt,try3.txt,better4.txt,better5.txt,try6.txt";
    size_t cMiniBatchSize = 16;
    size_t numberOfEpisodes = 10000; // ignore the function parameter for now until proper framework is in place
    size_t nrEpisodesToEpsilonZero = numberOfEpisodes / 4 * 3;
    size_t sizeExperience = 100000;
    float epsilon = 0.5;
    float gamma = 0.9;
    size_t agentVisionGridSize = 1;
    size_t agentVisionGridArea = agentVisionGridSize *2+1;
    agentVisionGridArea *= agentVisionGridArea;
    size_t opponentVisionGridSize = 1;
    size_t opponentVisionGridArea = opponentVisionGridSize *2+1;
    opponentVisionGridArea *= opponentVisionGridArea;
    globalRng = RandObj(275165314, -1, 1, sizeExperience);
    OpModellingType opModellingType = OpModellingType::ONEFORALL;
    ExpReplayParams expReplayParams{ .cSwapPeriod = 1000,
                                     .miniBatchSize = cMiniBatchSize,
                                     .sizeExperience = sizeExperience };
    AgentMonteCarloParams agentMonteCarloParams{ .maxNrSteps = 1, .nrRollouts = 5 };
    MLPParams agentMLP{ .sizes = { agentVisionGridArea *2+4, 200, 4 },
                        .learningRate = 0.001,
                        .regParam = -1,
                        .outputActivationFunc = ActivationFunction::LINEAR,
                        .miniBatchSize = cMiniBatchSize,
                        .randInit = false};
    MLPParams opponentMLP{ .sizes = { opponentVisionGridArea *3, 200, 4 },
                           .learningRate = 0.001,
                           .regParam  = -1,
                           .outputActivationFunc = ActivationFunction::SOFTMAX,
                           .miniBatchSize = cMiniBatchSize,
                           .randInit = false };
    Rewards rewards = { .normalReward = -0.01f,
                        .killedByOpponentReward = -1.0f,
                        .outOfBoundsReward = -0.01f,
                        .reachedGoalReward = 1.0f };
    SimStateParams simStateParams = { .traceSize = 6, .agentVisionGridSize = agentVisionGridSize,.opponentVisionGridSize = opponentVisionGridSize, .randomOpCoef = -1 };
    OpTrackParams kolsmirParams = { .pValueThreshold = 0.05, .minHistorySize = 10, .maxHistorySize = 20 };
    OpTrackParams pettittParams = { .pValueThreshold = 0.01, .minHistorySize = 10, .maxHistorySize = 20 };

    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent = std::make_unique<QERQueueLearning>(
        kolsmirParams, agentMonteCarloParams, agentMLP, opponentMLP, expReplayParams, numberOfEpisodes,
        nrEpisodesToEpsilonZero, OpModellingType::ONEFORALL, 0.5, gamma);
//        std::unique_ptr<Agent> agent =
//            std::make_unique<Sarsa>(kolsmirParams, agentMonteCarloParams, agentMLP, opponentMLP,
//                                               numberOfEpisodes,nrEpisodesToEpsilonZero,
//                                               OpModellingType::KOLSMIR,0.3,gamma);
    SimContainer simContainer{ files, agent.get(), rewards, simStateParams };
    agent->run();
    std::ofstream out{ "results/rewards04AFTER.txt" };
    std::vector<float> const &agentRewards = agent->getRewards();
    copy(agentRewards.begin(), agentRewards.end(), std::ostream_iterator<float>(out, "\n"));
    std::ofstream opponent{ "results/opponentPredictionLossesTwoDOUBLE.txt" };
    std::vector<float> const &opponentPred = agent->getOpponentPredictionLosses();
    copy(opponentPred.begin(), opponentPred.end(), std::ostream_iterator<float>(opponent, "\n"));
    std::ofstream opponentPerc{ "results/opponentPredictionPercentageTwoDOUBLE.txt" };
    std::vector<float> const &opponentPredPerc = agent->getOpponentCorrectPredictionPercentage();
    copy(opponentPredPerc.begin(), opponentPredPerc.end(), std::ostream_iterator<float>(opponentPerc, "\n"));
    std::ofstream trainLoss{"results/trainLoss2.txt"};
    std::vector<float> const &trainLossPerEp = agent->getLearningLosses();
    copy(trainLossPerEp.begin(), trainLossPerEp.end(),
             std::ostream_iterator<float>(trainLoss, "\n"));
    std::cout << "opponent prediction percentage: " << agent->getCorrectOpponentTypePredictionPercentage() << std::endl;
    std::cout << "nr of times killed by opponent: " << agent->getOpDeathPercentage() << std::endl;
    //    std::ofstream opponentLoss{"results/opponentFirstEpLoss.txt"};
    //    std::vector<float> const &opponentThisLoss = agent->getThisEpisodeLoss();
    //    copy(opponentThisLoss.begin(), opponentThisLoss.end(),
    //         std::ostream_iterator<float>(opponentLoss, "\n"));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "time in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
}
