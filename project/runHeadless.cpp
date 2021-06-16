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

HyperparamSpec loadHyperparameters(std::string const &file)
{
    std::ifstream in(file);
    if(not in)
    {
        throw std::runtime_error("could not open file");
    }
    HyperparamSpec hs;
    in >> hs;
    return hs;
}
void runHeadless(std::string const &file)
{
    auto begin = std::chrono::high_resolution_clock::now();
    //    std::cout.setstate(std::ios_base::failbit);
    HyperparamSpec hs = loadHyperparameters(file);
    size_t nrEpisodesToEpsilonZero = hs.numberOfEpisodes / 4 * 3;

    size_t agentVisionGridArea = hs.agentVisionGridSize *2+1;
    agentVisionGridArea *= agentVisionGridArea;

    size_t opponentVisionGridArea = hs.opponentVisionGridSize *2+1;
    opponentVisionGridArea *= opponentVisionGridArea;
    globalRng = RandObj(hs.seed, -1, 1, hs.sizeExperience);
    OpModellingType opModellingType = hs.opModellingType;
    ExpReplayParams expReplayParams{ .cSwapPeriod = hs.swapPeriod,
                                     .miniBatchSize = hs.miniBatchSize,
                                     .sizeExperience = hs.sizeExperience };
    AgentMonteCarloParams agentMonteCarloParams{ .maxNrSteps = hs.maxNrSteps, .nrRollouts = hs.nrRollouts };
    MLPParams agentMLP{ .sizes = { agentVisionGridArea *2+4, 200, 4 },
                        .learningRate = hs.agentLearningRate,
                        .regParam = hs.agentRegParam,
                        .outputActivationFunc = ActivationFunction::LINEAR,
                        .miniBatchSize = hs.miniBatchSize,
                        .randInit = false};
    MLPParams opponentMLP{ .sizes = { opponentVisionGridArea *3, 200, 4 },
                           .learningRate = hs.opponentLearningRate,
                           .regParam  = hs.opponentRegParam,
                           .outputActivationFunc = ActivationFunction::SOFTMAX,
                           .miniBatchSize = hs.miniBatchSize,
                           .randInit = false };
    Rewards rewards = { .normalReward = -0.01f,
                        .killedByOpponentReward = -1.0f,
                        .outOfBoundsReward = -0.01f,
                        .reachedGoalReward = 1.0f };
    SimStateParams simStateParams = { .traceSize = hs.traceSize, .agentVisionGridSize = hs.agentVisionGridSize,.opponentVisionGridSize = hs.opponentVisionGridSize, .randomOpCoef = hs.randomOpCoef };
    OpTrackParams opTrackParams = { .pValueThreshold = hs.pValueThreshold, .minHistorySize = hs.minHistorySize, .maxHistorySize = hs.maxHistorySize};
    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent;
    switch(hs.agentType)
    {

        case AgentType::SARSA:
            agent =
                std::make_unique<Sarsa>(opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP,
                                        hs.numberOfEpisodes,nrEpisodesToEpsilonZero,
                                        opModellingType,hs.epsilon,hs.gamma);
            break;
        case AgentType::DEEPQLEARNING:
            agent = std::make_unique<QERQueueLearning>(
                opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP, expReplayParams, hs.numberOfEpisodes,
                nrEpisodesToEpsilonZero, opModellingType, hs.epsilon, hs.gamma);
            break;
        case AgentType::DOUBLEDEEPQLEARNING:
            agent = std::make_unique<DQERQueueLearning>(
                opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP, expReplayParams, hs.numberOfEpisodes,
                nrEpisodesToEpsilonZero, opModellingType, hs.epsilon, hs.gamma);
            break;
    }


    SimContainer simContainer{ hs.files, agent.get(), rewards, simStateParams };
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
