#include "generateHyperparams.h"

#include "hyperparamSpec.h"
#include "opTypes.h"
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>

void generateHyperparams(std::string folder, std::string name, std::string newName)
{
    std::array<unsigned int, 8> seedRange = { 275165314,  3310202799, 1126433036, 2960205070,
                                              3500374631, 1192398765, 3162705734, 4003750407 };
    std::array<size_t, 2> miniBatchSizeRange = { 8,     16 };
    std::array<size_t, 2> sizeExperienceRange = { 10000, 100000 };
    std::array<float, 2> sarsaEpsilonRange = { 0.3f, 0.5f };
    std::array<float, 2> qLearningEpsilonRange = { 0.5f, 0.75f };
    std::array<float, 3> gammaRange = { 0.9f, 0.95f, 0.99f };
    std::array<size_t, 2> agentVisionGridSizeRange = { 1, 2 };
    std::array<size_t, 2> opponentVisionGridSizeRange = { 1, 2 };
    std::array<size_t, 2> swapPeriodRange = { 100, 1000 };
    std::array<size_t, 3> maxNrStepsRange = { 1, 3, 5 };
    std::array<size_t, 3> nrRolloutsRange = { 1, 3, 5 };
    std::array<float, 3> agentLearningRateRange = { 0.0001f, 0.0005f, 0.001f };
    std::array<float, 3> opponentLearningRateRange = { 0.0001f, 0.0005f, 0.001f };
    std::array<float, 4> pValueThresholdRange = { 0.025f, 0.05f, 0.1f, 0.2f };
    std::array<size_t, 2> minHistorySizeRange = { 8, 10 };
    std::array<size_t, 4> maxHistorySizeRange = { 20, 30, 50, 100 };
    std::array<OpModellingType, 3> opModellingTypeRange = {OpModellingType::ONEFORALL,OpModellingType::KOLSMIR,OpModellingType::NOTRAINPETTITT};
    std::array<AgentType, 3> agentTypeRange = {AgentType::SARSA, AgentType::DEEPQLEARNING, AgentType::DOUBLEDEEPQLEARNING};
    std::array<float, 2> randomCoefRange = {-1,0.2f};
    HyperparamSpec initialHs;
    std::ifstream in{ folder + name };
    if (not in)
    {
        throw std::runtime_error("could not open file");
    }
    in >> initialHs;
    std::array<float, 2> epsilonRange;
    if (static_cast<AgentType>(initialHs.agentType) == AgentType::SARSA)
        epsilonRange = sarsaEpsilonRange;
    else
        epsilonRange = qLearningEpsilonRange;
    auto genOpModelling = [&]()
    {
        for (size_t seedIdx = 0; seedIdx != seedRange.size(); ++seedIdx)
        {
            for (size_t pValueThresholdIdx = 0; pValueThresholdIdx != pValueThresholdRange.size(); ++pValueThresholdIdx)
            {
                for (size_t minHistorySizeIdx = 0; minHistorySizeIdx != minHistorySizeRange.size(); ++minHistorySizeIdx)
                {
                    for (size_t maxHistorySizeIdx = 0; maxHistorySizeIdx != maxHistorySizeRange.size();
                         ++maxHistorySizeIdx)
                    {
                        HyperparamSpec hs = { .seed = seedRange[seedIdx],
                                              .files = initialHs.files,
                                              .miniBatchSize = initialHs.miniBatchSize,
                                              .numberOfEpisodes = initialHs.numberOfEpisodes,
                                              .sizeExperience = initialHs.sizeExperience,
                                              .agentType = initialHs.agentType,
                                              .epsilon = initialHs.epsilon,
                                              .gamma = initialHs.gamma,
                                              .agentVisionGridSize = initialHs.agentVisionGridSize,
                                              .opponentVisionGridSize = initialHs.opponentVisionGridSize,
                                              .swapPeriod = initialHs.swapPeriod,
                                              .maxNrSteps = initialHs.maxNrSteps,
                                              .nrRollouts = initialHs.nrRollouts,
                                              .agentLearningRate = initialHs.agentLearningRate,
                                              .agentRegParam = initialHs.agentRegParam,
                                              .opponentLearningRate = initialHs.opponentLearningRate,
                                              .opponentRegParam = initialHs.opponentRegParam,
                                              .traceSize = initialHs.traceSize,
                                              .randomOpCoef = initialHs.randomOpCoef,
                                              .opModellingType = initialHs.opModellingType,
                                              .pValueThreshold = pValueThresholdRange[pValueThresholdIdx],
                                              .minHistorySize = minHistorySizeRange[minHistorySizeIdx],
                                              .maxHistorySize = maxHistorySizeRange[maxHistorySizeIdx]

                        };
                        std::ofstream out{ "bestForThreeSarsaOpModeling/" +
                                           std::to_string(seedIdx) + std::to_string(pValueThresholdIdx) +
                                           std::to_string(minHistorySizeIdx) + std::to_string(maxHistorySizeIdx) +
                                           +"seed_pValueThreshold_minHistorySize_maxHistorySize_" + newName };
                        out << hs;
                    }
                }
            }
        }
    };
    auto genAgent = [&]()
    {
        for (size_t seedIdx = 0; seedIdx != seedRange.size(); ++seedIdx)
        {
            for (size_t epsilonIdx = 0; epsilonIdx != epsilonRange.size(); ++epsilonIdx)
            {
                for (size_t gammaIdx = 0; gammaIdx != gammaRange.size(); ++gammaIdx)
                {
                    HyperparamSpec hs = { .seed = seedRange[seedIdx],
                                          .files = initialHs.files,
                                          .miniBatchSize = initialHs.miniBatchSize,
                                          .numberOfEpisodes = initialHs.numberOfEpisodes,
                                          .sizeExperience = initialHs.sizeExperience,
                                          .agentType = initialHs.agentType,
                                          .epsilon = epsilonRange[epsilonIdx],
                                          .gamma = gammaRange[gammaIdx],
                                          .agentVisionGridSize = initialHs.agentVisionGridSize,
                                          .opponentVisionGridSize = initialHs.opponentVisionGridSize,
                                          .swapPeriod = initialHs.swapPeriod,
                                          .maxNrSteps = initialHs.maxNrSteps,
                                          .nrRollouts = initialHs.nrRollouts,
                                          .agentLearningRate = initialHs.agentLearningRate,
                                          .agentRegParam = initialHs.agentRegParam,
                                          .opponentLearningRate = initialHs.opponentLearningRate,
                                          .opponentRegParam = initialHs.opponentRegParam,
                                          .traceSize = initialHs.traceSize,
                                          .randomOpCoef = initialHs.randomOpCoef,
                                          .opModellingType = initialHs.opModellingType,
                                          .pValueThreshold = initialHs.pValueThreshold,
                                          .minHistorySize = initialHs.minHistorySize,
                                          .maxHistorySize = initialHs.maxHistorySize

                    };
                    std::ofstream out{ "seed_epsilon_gamma/" + std::to_string(seedIdx) + std::to_string(epsilonIdx) +
                                       std::to_string(gammaIdx) + +"seed_epsilon_gamma_" + newName };
                    out << hs;
                }
            }
        }
    };
    auto genExperienceReplay = [&]()
    {
        for (size_t seedIdx = 0; seedIdx != seedRange.size(); ++seedIdx)
        {
            for (size_t miniBatchSizeIdx = 0; miniBatchSizeIdx != miniBatchSizeRange.size(); ++miniBatchSizeIdx)
            {
                for (size_t sizeExperienceIdx = 0; sizeExperienceIdx != sizeExperienceRange.size(); ++sizeExperienceIdx)
                {
                    for (size_t swapPeriodIdx = 0; swapPeriodIdx != swapPeriodRange.size(); ++swapPeriodIdx)
                    {
                        HyperparamSpec hs = { .seed = seedRange[seedIdx],
                                              .files = initialHs.files,
                                              .miniBatchSize = miniBatchSizeRange[miniBatchSizeIdx],
                                              .numberOfEpisodes = initialHs.numberOfEpisodes,
                                              .sizeExperience = sizeExperienceRange[sizeExperienceIdx],
                                              .agentType = initialHs.agentType,
                                              .epsilon = initialHs.epsilon,
                                              .gamma = initialHs.gamma,
                                              .agentVisionGridSize = initialHs.agentVisionGridSize,
                                              .opponentVisionGridSize = initialHs.opponentVisionGridSize,
                                              .swapPeriod = swapPeriodRange[swapPeriodIdx],
                                              .maxNrSteps = initialHs.maxNrSteps,
                                              .nrRollouts = initialHs.nrRollouts,
                                              .agentLearningRate = initialHs.agentLearningRate,
                                              .agentRegParam = initialHs.agentRegParam,
                                              .opponentLearningRate = initialHs.opponentLearningRate,
                                              .opponentRegParam = initialHs.opponentRegParam,
                                              .traceSize = initialHs.traceSize,
                                              .randomOpCoef = initialHs.randomOpCoef,
                                              .opModellingType = initialHs.opModellingType,
                                              .pValueThreshold = initialHs.pValueThreshold,
                                              .minHistorySize = initialHs.minHistorySize,
                                              .maxHistorySize = initialHs.maxHistorySize

                        };
                        std::ofstream out{ "seed_miniBatchSize_sizeExperience_swapPeriod/" + std::to_string(seedIdx) +
                                           std::to_string(miniBatchSizeIdx) + std::to_string(sizeExperienceIdx) +
                                           std::to_string(swapPeriodIdx) +
                                           +"seed_miniBatchSize_sizeExperience_swapPeriod_" + newName };
                        out << hs;
                    }
                }
            }
        }
    };
    auto genMLPAgent = [&]()
    {
        for (size_t seedIdx = 0; seedIdx != seedRange.size(); ++seedIdx)
        {
            for (size_t agentVisionGridSizeIdx = 0; agentVisionGridSizeIdx != agentVisionGridSizeRange.size();
                 ++agentVisionGridSizeIdx)
            {
                for (size_t agentLearningRateIdx = 0; agentLearningRateIdx != agentLearningRateRange.size();
                     ++agentLearningRateIdx)
                {
                    HyperparamSpec hs = { .seed = seedRange[seedIdx],
                                          .files = initialHs.files,
                                          .miniBatchSize = initialHs.miniBatchSize,
                                          .numberOfEpisodes = initialHs.numberOfEpisodes,
                                          .sizeExperience = initialHs.sizeExperience,
                                          .agentType = initialHs.agentType,
                                          .epsilon = initialHs.epsilon,
                                          .gamma = initialHs.gamma,
                                          .agentVisionGridSize = agentVisionGridSizeRange[agentVisionGridSizeIdx],
                                          .opponentVisionGridSize = initialHs.opponentVisionGridSize,
                                          .swapPeriod = initialHs.swapPeriod,
                                          .maxNrSteps = initialHs.maxNrSteps,
                                          .nrRollouts = initialHs.nrRollouts,
                                          .agentLearningRate = agentLearningRateRange[agentLearningRateIdx],
                                          .agentRegParam = initialHs.agentRegParam,
                                          .opponentLearningRate = initialHs.opponentLearningRate,
                                          .opponentRegParam = initialHs.opponentRegParam,
                                          .traceSize = initialHs.traceSize,
                                          .randomOpCoef = initialHs.randomOpCoef,
                                          .opModellingType = initialHs.opModellingType,
                                          .pValueThreshold = initialHs.pValueThreshold,
                                          .minHistorySize = initialHs.minHistorySize,
                                          .maxHistorySize = initialHs.maxHistorySize
                    };
                    std::ofstream out{ "test/" + std::to_string(seedIdx) +
                                       std::to_string(agentVisionGridSizeIdx) + std::to_string(agentLearningRateIdx) +
                                       +"seed_agentVisionGridSize_agentLearningRate_" + newName };
                    out << hs;
                }
            }
        }
    };
    auto genMLPOpponent = [&]()
    {
        for (size_t seedIdx = 0; seedIdx != seedRange.size(); ++seedIdx)
        {
            for (size_t opponentVisionGridSizeIdx = 0; opponentVisionGridSizeIdx != opponentVisionGridSizeRange.size();
                 ++opponentVisionGridSizeIdx)
            {
                for (size_t opponentLearningRateIdx = 0; opponentLearningRateIdx != opponentLearningRateRange.size();
                     ++opponentLearningRateIdx)
                {
                    HyperparamSpec hs = { .seed = seedRange[seedIdx],
                                          .files = initialHs.files,
                                          .miniBatchSize = initialHs.miniBatchSize,
                                          .numberOfEpisodes = initialHs.numberOfEpisodes,
                                          .sizeExperience = initialHs.sizeExperience,
                                          .agentType = initialHs.agentType,
                                          .epsilon = initialHs.epsilon,
                                          .gamma = initialHs.gamma,
                                          .agentVisionGridSize = initialHs.agentVisionGridSize,
                                          .opponentVisionGridSize =
                                              opponentVisionGridSizeRange[opponentVisionGridSizeIdx],
                                          .swapPeriod = initialHs.swapPeriod,
                                          .maxNrSteps = initialHs.maxNrSteps,
                                          .nrRollouts = initialHs.nrRollouts,
                                          .agentLearningRate = initialHs.agentLearningRate,
                                          .agentRegParam = initialHs.agentRegParam,
                                          .opponentLearningRate = opponentLearningRateRange[opponentLearningRateIdx],
                                          .opponentRegParam = initialHs.opponentRegParam,
                                          .traceSize = initialHs.traceSize,
                                          .randomOpCoef = initialHs.randomOpCoef,
                                          .opModellingType = initialHs.opModellingType,
                                          .pValueThreshold = initialHs.pValueThreshold,
                                          .minHistorySize = initialHs.minHistorySize,
                                          .maxHistorySize = initialHs.maxHistorySize

                    };
                    std::ofstream out{ "newop_seed_opponentVisionGridSize_opponentLearningRate/" + std::to_string(seedIdx) +
                                       std::to_string(opponentVisionGridSizeIdx) +
                                       std::to_string(opponentLearningRateIdx) +
                                       +"seed_opponentVisionGridSize_opponentLearningRate_" + newName };
                    out << hs;
                }
            }
        }
    };
    auto genMonteCarlo = [&]()
    {
        for (size_t seedIdx = 0; seedIdx != seedRange.size(); ++seedIdx)
        {
            for (size_t maxNrStepsIdx = 0; maxNrStepsIdx != maxNrStepsRange.size(); ++maxNrStepsIdx)
            {
                for (size_t nrRolloutsIdx = 0; nrRolloutsIdx != nrRolloutsRange.size(); ++nrRolloutsIdx)
                {
                    HyperparamSpec hs = { .seed = seedRange[seedIdx],
                                          .files = initialHs.files,
                                          .miniBatchSize = initialHs.miniBatchSize,
                                          .numberOfEpisodes = initialHs.numberOfEpisodes,
                                          .sizeExperience = initialHs.sizeExperience,
                                          .agentType = initialHs.agentType,
                                          .epsilon = initialHs.epsilon,
                                          .gamma = initialHs.gamma,
                                          .agentVisionGridSize = initialHs.agentVisionGridSize,
                                          .opponentVisionGridSize = initialHs.opponentVisionGridSize,
                                          .swapPeriod = initialHs.swapPeriod,
                                          .maxNrSteps = maxNrStepsRange[maxNrStepsIdx],
                                          .nrRollouts = nrRolloutsRange[nrRolloutsIdx],
                                          .agentLearningRate = initialHs.agentLearningRate,
                                          .agentRegParam = initialHs.agentRegParam,
                                          .opponentLearningRate = initialHs.opponentLearningRate,
                                          .opponentRegParam = initialHs.opponentRegParam,
                                          .traceSize = initialHs.traceSize,
                                          .randomOpCoef = initialHs.randomOpCoef,
                                          .opModellingType = initialHs.opModellingType,
                                          .pValueThreshold = initialHs.pValueThreshold,
                                          .minHistorySize = initialHs.minHistorySize,
                                          .maxHistorySize = initialHs.maxHistorySize

                    };
                    std::ofstream out{ "seed_maxNrSteps_nrRollouts/" + std::to_string(seedIdx) +
                                       std::to_string(maxNrStepsIdx) + std::to_string(nrRolloutsIdx) +
                                       +"seed_maxNrSteps_nrRollouts_" + newName };
                    out << hs;
                }
            }
        }
    };
    auto genGivenRandom = [&]()
    {
        for (size_t seedIdx = 0; seedIdx != seedRange.size(); ++seedIdx)
        {
            HyperparamSpec hs = { .seed = seedRange[seedIdx],
                                  .files = initialHs.files,
                                  .miniBatchSize = initialHs.miniBatchSize,
                                  .numberOfEpisodes = initialHs.numberOfEpisodes,
                                  .sizeExperience = initialHs.sizeExperience,
                                  .agentType = initialHs.agentType,
                                  .epsilon = initialHs.epsilon,
                                  .gamma = initialHs.gamma,
                                  .agentVisionGridSize = initialHs.agentVisionGridSize,
                                  .opponentVisionGridSize = initialHs.opponentVisionGridSize,
                                  .swapPeriod = initialHs.swapPeriod,
                                  .maxNrSteps = initialHs.maxNrSteps,
                                  .nrRollouts = initialHs.nrRollouts,
                                  .agentLearningRate = initialHs.agentLearningRate,
                                  .agentRegParam = initialHs.agentRegParam,
                                  .opponentLearningRate = initialHs.opponentLearningRate,
                                  .opponentRegParam = initialHs.opponentRegParam,
                                  .traceSize = initialHs.traceSize,
                                  .randomOpCoef = initialHs.randomOpCoef,
                                  .opModellingType = initialHs.opModellingType,
                                  .pValueThreshold = initialHs.pValueThreshold,
                                  .minHistorySize = initialHs.minHistorySize,
                                  .maxHistorySize = initialHs.maxHistorySize

            };
            std::ofstream out{ "MONTECARLORANDOMHYPERPARAMS/" + std::to_string(seedIdx) + "seed_givenRandom_" + newName };
            out << hs;
        }
    };
    auto genInitialSetups = [&]()
    {
      for (size_t randCoefIdx = 0; randCoefIdx != randomCoefRange.size(); ++randCoefIdx)
      {
          for (size_t agentTypeIdx = 0; agentTypeIdx != agentTypeRange.size(); ++agentTypeIdx)
          {
              for (size_t opModellingTypeIdx = 0; opModellingTypeIdx != opModellingTypeRange.size(); ++opModellingTypeIdx)
              {
                  HyperparamSpec hs = { .seed = initialHs.seed,
                      .files = initialHs.files,
                      .miniBatchSize = initialHs.miniBatchSize,
                      .numberOfEpisodes = initialHs.numberOfEpisodes,
                      .sizeExperience = initialHs.sizeExperience,
                      .agentType = agentTypeRange[agentTypeIdx],
                      .epsilon = initialHs.epsilon,
                      .gamma = initialHs.gamma,
                      .agentVisionGridSize = initialHs.agentVisionGridSize,
                      .opponentVisionGridSize = initialHs.opponentVisionGridSize,
                      .swapPeriod = initialHs.swapPeriod,
                      .maxNrSteps = initialHs.maxNrSteps,
                      .nrRollouts = initialHs.nrRollouts,
                      .agentLearningRate = initialHs.agentLearningRate,
                      .agentRegParam = initialHs.agentRegParam,
                      .opponentLearningRate = initialHs.opponentLearningRate,
                      .opponentRegParam = initialHs.opponentRegParam,
                      .traceSize = initialHs.traceSize,
                      .randomOpCoef = randomCoefRange[randCoefIdx],
                      .opModellingType = opModellingTypeRange[opModellingTypeIdx],
                      .pValueThreshold = initialHs.pValueThreshold,
                      .minHistorySize = initialHs.minHistorySize,
                      .maxHistorySize = initialHs.maxHistorySize

                  };
                  std::ofstream out{ "randCoef_agentType_opModellingType/" + std::to_string(randCoefIdx) +
                                     std::to_string(agentTypeIdx) + std::to_string(opModellingTypeIdx) +
                                     + ".txt" };
                  out << hs;
              }
          }
      }
    };
    genGivenRandom();
}