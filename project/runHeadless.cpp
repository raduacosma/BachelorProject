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
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>

RandObj globalRng;

HyperparamSpec loadHyperparameters(std::string const &file)
{
    std::ifstream in(file);
    if (not in)
    {
        throw std::runtime_error("could not open file");
    }
    HyperparamSpec hs;
    in >> hs;
    return hs;
}
void writeFullResults(std::unique_ptr<Agent> &agent)
{
    std::ofstream out{ "results/rewards04AFTER.txt" };
    std::vector<float> const &agentRewards = agent->getRewards();
    copy(agentRewards.begin(), agentRewards.end(), std::ostream_iterator<float>(out, "\n"));
    std::ofstream opponent{ "results/opponentPredictionLossesTwoDOUBLE.txt" };
    std::vector<float> const &opponentPred = agent->getOpponentPredictionLosses();
    copy(opponentPred.begin(), opponentPred.end(), std::ostream_iterator<float>(opponent, "\n"));
    std::ofstream opponentPerc{ "results/opponentPredictionPercentageTwoDOUBLE.txt" };
    std::vector<float> const &opponentPredPerc = agent->getOpponentCorrectPredictionPercentage();
    copy(opponentPredPerc.begin(), opponentPredPerc.end(), std::ostream_iterator<float>(opponentPerc, "\n"));
    std::ofstream trainLoss{ "results/trainLoss2.txt" };
    std::vector<float> const &trainLossPerEp = agent->getLearningLosses();
    copy(trainLossPerEp.begin(), trainLossPerEp.end(), std::ostream_iterator<float>(trainLoss, "\n"));
    std::cout << "opponent prediction percentage: " << agent->getCorrectOpponentTypePredictionPercentage() << std::endl;
    std::cout << "nr of times killed by opponent: " << agent->getOpDeathPercentage() << std::endl;
    //    std::ofstream opponentLoss{"results/opponentFirstEpLoss.txt"};
    //    std::vector<float> const &opponentThisLoss = agent->getThisEpisodeLoss();
    //    copy(opponentThisLoss.begin(), opponentThisLoss.end(),
    //         std::ostream_iterator<float>(opponentLoss, "\n"));
}

void writeSummaryResults(std::unique_ptr<Agent> &agent, std::string const &fileName, size_t nrEpisodesToEpsilonZero, size_t numberOfEpisodes, long totalTime)
{
    std::cout<<fileName<<'\n';
//    std::ofstream out{"summaryone_"+fileName};
    std::vector<float> const &agentRewards = agent->getRewards();
    double rewardSum= std::accumulate(agentRewards.begin() + nrEpisodesToEpsilonZero, agentRewards.begin()+numberOfEpisodes,0.0 );
    std::cout<<"rewardsMeanLast\n"<<rewardSum/(numberOfEpisodes-nrEpisodesToEpsilonZero)<<'\n';
    std::vector<float> const &agentPredictions = agent->getOpponentCorrectPredictionPercentage();
    double opPredSum = std::accumulate(agentPredictions.begin() + nrEpisodesToEpsilonZero, agentPredictions.begin()+numberOfEpisodes,0.0);
    std::cout<<"opPredMeanLast\n"<<opPredSum/(numberOfEpisodes-nrEpisodesToEpsilonZero)<<'\n';
    std::vector<float> const &foundOpPred = agent->getOpponentFoundCorrectPredictionPercentage();
    double foundOpSum = std::accumulate(foundOpPred.begin() + nrEpisodesToEpsilonZero, foundOpPred.begin()+numberOfEpisodes,0.0);
    std::cout<<"opponentFoundPredMeanLast\n"<<foundOpSum/static_cast<double>(numberOfEpisodes-nrEpisodesToEpsilonZero)<<'\n';
    std::vector<size_t> const &killedPerEp = agent->getOpDeathsPerEp();
    size_t killedSum = std::accumulate(killedPerEp.begin() + nrEpisodesToEpsilonZero, killedPerEp.begin()+numberOfEpisodes,0ul);
    std::cout<<"killedByOpponentMeanLast\n"<<killedSum/static_cast<double>(numberOfEpisodes-nrEpisodesToEpsilonZero)<<'\n';
    std::cout << "opponent recognition percentage\n" << agent->getCorrectOpponentTypePredictionPercentage() << '\n';
    std::cout << "predicted nr of opponents\n" << agent->getPredictedNrOfOpponents()<<'\n';
    for(auto const &opIdx:agent->opChoiceMatrix)
    {
        std::cout<<"Opponent idx: "<<opIdx.first<<std::endl;
        for (auto const &chosenIdx : opIdx.second)
        {
            std::cout<<"Chosen Opponent: "<<chosenIdx.first<<" "<<chosenIdx.second<<std::endl;
        }
    }
    std::cout<<"time in ms\n"<<totalTime<<'\n';


}

void writeFullEpHistory(std::unique_ptr<Agent> &agent, std::string const &fileName)
{
    std::cout<<"actualOpType,predOpType,rewards,opPredPerc,foundOpPredPerc,killedByOpPerc\n";
    auto const &actualOpType = agent->getActualOpponentType();
    auto const &predOpType = agent->getPredictedOpponentType();
    auto const &rewards = agent->getRewards();
    auto const &opPredPerc = agent->getOpponentCorrectPredictionPercentage();
    auto const &foundOpPredPerc = agent->getOpponentFoundCorrectPredictionPercentage();
    auto const &killedByOpPerc = agent->getOpDeathsPerEp();
    for(size_t idx=0; idx!=opPredPerc.size();++idx)
    {
        std::cout<<actualOpType[idx]<<','<<predOpType[idx]<<','<<rewards[idx]<<','<<opPredPerc[idx]
            <<','<<foundOpPredPerc[idx]<<','<<killedByOpPerc[idx]<<'\n';
    }
    for(size_t idx = opPredPerc.size();idx!=actualOpType.size();++idx)
    {
        std::cout<<actualOpType[idx]<<','<<predOpType[idx]<<",NA,NA,NA,NA\n";
    }
    std::cout << "opponent recognition percentage\n" << agent->getCorrectOpponentTypePredictionPercentage() << '\n';
    std::cout << "predicted nr of opponents\n" << agent->getPredictedNrOfOpponents()<<'\n';
    std::cout<<fileName<<'\n';
}
void runHeadless(std::string const &file)
{
    auto begin = std::chrono::high_resolution_clock::now();
    //    std::cout.setstate(std::ios_base::failbit);
    HyperparamSpec hs = loadHyperparameters(file);
    size_t nrEpisodesToEpsilonZero = hs.numberOfEpisodes / 4 * 3;

    size_t agentVisionGridArea = hs.agentVisionGridSize * 2 + 1;
    agentVisionGridArea *= agentVisionGridArea;

    size_t opponentVisionGridArea = hs.opponentVisionGridSize * 2 + 1;
    opponentVisionGridArea *= opponentVisionGridArea;
    globalRng = RandObj(hs.seed, -1, 1, hs.sizeExperience);
    OpModellingType opModellingType = hs.opModellingType;
    ExpReplayParams expReplayParams{ .cSwapPeriod = hs.swapPeriod,
                                     .miniBatchSize = hs.miniBatchSize,
                                     .sizeExperience = hs.sizeExperience };
    AgentMonteCarloParams agentMonteCarloParams{ .maxNrSteps = hs.maxNrSteps, .nrRollouts = hs.nrRollouts };
    MLPParams agentMLP{ .sizes = { agentVisionGridArea * 2 + 4, 200, 4 },
                        .learningRate = hs.agentLearningRate,
                        .regParam = hs.agentRegParam,
                        .outputActivationFunc = ActivationFunction::LINEAR,
                        .miniBatchSize = hs.miniBatchSize,
                        .randInit = false };
    MLPParams opponentMLP{ .sizes = { opponentVisionGridArea * 3, 200, 4 },
                           .learningRate = hs.opponentLearningRate,
                           .regParam = hs.opponentRegParam,
                           .outputActivationFunc = ActivationFunction::SOFTMAX,
                           .miniBatchSize = hs.miniBatchSize,
                           .randInit = false };
    Rewards rewards = {
        .normalReward = -0.01f, .killedByOpponentReward = -1.0f, .outOfBoundsReward = -0.01f, .reachedGoalReward = 1.0f
    };
    SimStateParams simStateParams = { .traceSize = hs.traceSize,
                                      .agentVisionGridSize = hs.agentVisionGridSize,
                                      .opponentVisionGridSize = hs.opponentVisionGridSize,
                                      .randomOpCoef = hs.randomOpCoef };
    OpTrackParams opTrackParams = { .pValueThreshold = hs.pValueThreshold,
                                    .minHistorySize = hs.minHistorySize,
                                    .maxHistorySize = hs.maxHistorySize };
    // could also use stack but meh, this way is more certain
    std::unique_ptr<Agent> agent;
    switch (hs.agentType)
    {

        case AgentType::SARSA:
            agent = std::make_unique<Sarsa>(opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP,
                                            hs.numberOfEpisodes, nrEpisodesToEpsilonZero, opModellingType, hs.epsilon,
                                            hs.gamma);
            break;
        case AgentType::DEEPQLEARNING:
            agent = std::make_unique<QERQueueLearning>(opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP,
                                                       expReplayParams, hs.numberOfEpisodes, nrEpisodesToEpsilonZero,
                                                       opModellingType, hs.epsilon, hs.gamma);
            break;
        case AgentType::DOUBLEDEEPQLEARNING:
            agent = std::make_unique<DQERQueueLearning>(opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP,
                                                        expReplayParams, hs.numberOfEpisodes, nrEpisodesToEpsilonZero,
                                                        opModellingType, hs.epsilon, hs.gamma);
            break;
    }

    SimContainer simContainer{ hs.files, agent.get(), rewards, simStateParams };
    agent->run();
//    writeFullResults(agent);
    auto end = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
//    std::cout << "time in ms: " << totalTime;
    writeFullEpHistory(agent,file);
//    writeSummaryResults(agent,file,nrEpisodesToEpsilonZero,hs.numberOfEpisodes,totalTime);
}
