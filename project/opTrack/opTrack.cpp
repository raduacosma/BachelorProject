#include "opTrack.h"
#include "../agent/agent.h"
#include "../kolsmir/kolsmir.h"
#include "../pettitt/pettitt.h"
#include <cassert>
#include <iostream>

OpTrack::OpTrack(double pPValueThreshold, size_t pMinHistorySize, size_t pMaxHistorySize)
    : pValueThreshold(pPValueThreshold), minHistorySize(pMinHistorySize), maxHistorySize(pMaxHistorySize)
{
}
void OpTrack::destroyRandomKolsmir(Agent &agent)
{
    agent.opList.pop_back();
    agent.opLosses.pop_back();
    opListLossHistory.pop_back();
    opListStateHistory.pop_back();
}
void OpTrack::destroyRandomPettitt(Agent &agent)
{
    agent.opList.pop_back();
    agent.opLosses.pop_back();
    opDequeLossHistory.pop_back();
}
void OpTrack::destroyRandomNoTrainPettitt(Agent &agent)
{
    agent.opList.pop_back();
    agent.opLosses.pop_back();
    opListLossHistory.pop_back();
    opDequeStateHistory.pop_back();
}
void OpTrack::commonOpInit(Agent &agent)
{
    opHistoryCounter = 0;
    foundOpModel = false;
    currOpListLossHistory.resize(0);
    currOpListStateHistory.resize(0);
    if (firstTime)
    {
        firstTime = false;
        currOpListLossHistory.reserve(maxHistorySize);
        currOpListStateHistory.reserve(maxHistorySize);
        return;
    }
    agent.opLosses.push_back(0.0f);
    agent.opList.emplace_back(agent.opMLPParams.sizes, agent.opMLPParams.learningRate, agent.opMLPParams.regParam,
                              agent.opMLPParams.outputActivationFunc, agent.opMLPParams.miniBatchSize,
                              agent.opMLPParams.randInit);
    agent.currOp = agent.opList.size() - 1;
}
void OpTrack::noTrainPettittOpInit(Agent &agent)
{
    if (not firstTime)
    {

        if (opDequeStateHistory[agent.currOp].size() < minHistorySize or not foundOpModel)
        {
            destroyRandomNoTrainPettitt(agent);
        }
        else
        {
            // the current opponent will be switched now, so it has fully learned,
            // therefore recompute the losses with the new opponent in mind

            std::deque<OpExperience> &opStateRef = opDequeStateHistory[agent.currOp];
            std::vector<double> &opLossRef = opListLossHistory[agent.currOp];
            MLP &currOpRef = agent.opList[agent.currOp];
            opLossRef.resize(0);
            std::transform(opStateRef.cbegin(), opStateRef.cend(), std::back_inserter(opLossRef),
                           [&](OpExperience const &currExperience)
                           {
                               return currOpRef.predictWithLoss(currExperience.lastState, currExperience.newState);
                           });
        }
    }

    opDequeStateHistory.emplace_back();
    opListLossHistory.emplace_back();
    opListLossHistory.back().reserve(maxHistorySize);
    commonOpInit(agent);
}
void OpTrack::kolsmirOpInit(Agent &agent)
{
    if (not firstTime)
    {
        if (opListStateHistory[agent.currOp].size() < minHistorySize or not foundOpModel)
        {
            destroyRandomKolsmir(agent);
        }
        else
        {
            // the current opponent will be switched now, so it has fully learned,
            // therefore recompute the losses with the new opponent in mind

            std::vector<OpExperience> &opStateRef = opListStateHistory[agent.currOp];
            std::vector<double> &opLossRef = opListLossHistory[agent.currOp];
            MLP &currOpRef = agent.opList[agent.currOp];
            opLossRef.resize(0);
            std::transform(opStateRef.cbegin(), opStateRef.cend(), std::back_inserter(opLossRef),
                           [&](OpExperience const &currExperience)
                           {
                               return currOpRef.predictWithLoss(currExperience.lastState, currExperience.newState);
                           });
            std::sort(opLossRef.begin(), opLossRef.end());
        }
    }

    opListStateHistory.emplace_back();
    opListStateHistory.back().reserve(maxHistorySize);
    opListLossHistory.emplace_back();
    opListLossHistory.back().reserve(maxHistorySize);
    commonOpInit(agent);
}
void OpTrack::pettittOpInit(Agent &agent)
{
    if (not firstTime and (opDequeLossHistory[agent.currOp].size() < minHistorySize or not foundOpModel))
    {
        destroyRandomPettitt(agent);
    }
    if (firstTime)
        opCopies = std::vector<MLP>();
    else
        opCopies = agent.opList;
    opDequeLossHistory.emplace_back();
    commonOpInit(agent);
}
void OpTrack::pettittOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                                float loss)
{
    if (not foundOpModel and opHistoryCounter < minHistorySize)
    {
        currOpListStateHistory.push_back({ lastState, newState });
        opDequeLossHistory[agent.currOp].push_back(loss);
        ++opHistoryCounter;
        return;
    }
    if (not foundOpModel and opHistoryCounter >= minHistorySize)
    {
        double maxProb = -1;
        int maxProbIdx = -1;
        std::vector<double> maxOpListLossHistory;
        if (!opCopies.empty())
        {
            for (size_t opIdx = 0, opSz = opCopies.size(); opIdx != opSz; ++opIdx)
            {
                MLP &currDoneOp = opCopies[opIdx];
                currOpListLossHistory.resize(0);
                for (auto const &currState : currOpListStateHistory)
                {
                    currOpListLossHistory.push_back(currDoneOp.train(currState.lastState, currState.newState));
                }
                auto [prob, U, K] = Pettitt{}.testGivenK(opDequeLossHistory[opIdx], currOpListLossHistory,
                                                         opDequeLossHistory[opIdx].size() - 1);
                if (maxProb < prob)
                {
                    maxProb = prob;
                    maxProbIdx = opIdx;
                    maxOpListLossHistory = currOpListLossHistory;
                }
            }
        }
        foundOpModel = true;
        if (maxProbIdx != -1 and maxProb > pValueThreshold)
        {
            // this corresponds to removing the random MLP that was created
            destroyRandomPettitt(agent);
            agent.currOp = maxProbIdx;
            std::deque<double> &opLossRef = opDequeLossHistory[agent.currOp];
            // now train the currOp on the examples since they belong to it and add them to the history if maxHistory
            // was not achieved
            for (size_t idx = 0, sz = maxOpListLossHistory.size(); idx != sz; ++idx)
            {
                if (opLossRef.size() < maxHistorySize)
                    opLossRef.push_back(maxOpListLossHistory[idx]);
                else
                {
                    opLossRef.pop_front();
                    opLossRef.push_back(maxOpListLossHistory[idx]);
                }
            }
            // replace the old MLP with the new copy that was trained on the current examples
            agent.opList[agent.currOp] = opCopies[agent.currOp];
            // update loss for the newly chosen agent
            loss = agent.opList[agent.currOp].train(lastState, newState);
        }
        updateCorrectPercentage(agent);
    }
    if (foundOpModel and opDequeLossHistory[agent.currOp].size() < maxHistorySize)
        opDequeLossHistory[agent.currOp].push_back(loss);
    else if (foundOpModel and opDequeLossHistory[agent.currOp].size() >= maxHistorySize)
    {
        opDequeLossHistory[agent.currOp].pop_front();
        opDequeLossHistory[agent.currOp].push_back(loss);
    }
}
void OpTrack::kolsmirOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                                float loss)
{
    if (not foundOpModel and opHistoryCounter < minHistorySize)
    {
        currOpListStateHistory.push_back({ lastState, newState });
        opListStateHistory[agent.currOp].push_back({ lastState, newState });
        ++opHistoryCounter;
    }
    if (not foundOpModel and opHistoryCounter >= minHistorySize)
    {
        double maxProb = -1;
        int maxProbIdx = -1;
        for (size_t opIdx = 0, opSz = agent.opList.size() - 1; opIdx != opSz; ++opIdx)
        {
            MLP &currDoneOp = agent.opList[opIdx];
            currOpListLossHistory.resize(0);
            std::transform(currOpListStateHistory.cbegin(), currOpListStateHistory.cend(),
                           std::back_inserter(currOpListLossHistory),
                           [&](OpExperience const &currExperience)
                           {
                               return currDoneOp.predictWithLoss(currExperience.lastState, currExperience.newState);
                           });
            std::sort(currOpListLossHistory.begin(), currOpListLossHistory.end());
            // both arrays need to be sorted, opListLossHistory[opIdx] should already be sorted from init
            auto [prob, rdmax] = KolSmir{}.test(opListLossHistory[opIdx].size(), opListLossHistory[opIdx].data(),
                                                currOpListLossHistory.size(), currOpListLossHistory.data());
            if (maxProb < prob)
            {
                maxProb = prob;
                maxProbIdx = opIdx;
            }
        }
        foundOpModel = true;
        if (maxProbIdx != -1 and maxProb > pValueThreshold)
        {
            // this corresponds to removing the random MLP that was created
            destroyRandomKolsmir(agent);
            agent.currOp = maxProbIdx;
            std::vector<OpExperience> &opStateRef = opListStateHistory[agent.currOp];
            MLP &currOpRef = agent.opList[agent.currOp];
            // now train the currOp on the examples since they belong to it and add them to the history if maxHistory
            // was not achieved
            for (size_t idx = 0, sz = currOpListStateHistory.size(); idx != sz; ++idx)
            {
                OpExperience const &currExperience = currOpListStateHistory[idx];
                currOpRef.train(currExperience.lastState, currExperience.newState);
                if (opStateRef.size() < maxHistorySize)
                    opStateRef.push_back(currExperience);
            }
        }
        updateCorrectPercentage(agent);
        return;
    }
    if (foundOpModel and opListStateHistory[agent.currOp].size() < maxHistorySize)
        opListStateHistory[agent.currOp].push_back({ lastState, newState });
}
void OpTrack::noTrainPettittOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                                       float loss)
{
    if (not foundOpModel and opHistoryCounter < minHistorySize)
    {
        currOpListStateHistory.push_back({ lastState, newState });
        opDequeStateHistory[agent.currOp].push_back({ lastState, newState });
        ++opHistoryCounter;
    }
    if (not foundOpModel and opHistoryCounter >= minHistorySize)
    {
        double maxProb = -1;
        int maxProbIdx = -1;
        for (size_t opIdx = 0, opSz = agent.opList.size() - 1; opIdx != opSz; ++opIdx)
        {
            MLP &currDoneOp = agent.opList[opIdx];
            currOpListLossHistory.resize(0);
            std::transform(currOpListStateHistory.cbegin(), currOpListStateHistory.cend(),
                           std::back_inserter(currOpListLossHistory),
                           [&](OpExperience const &currExperience)
                           {
                               return currDoneOp.predictWithLoss(currExperience.lastState, currExperience.newState);
                           });
            auto [prob, U, K] = Pettitt{}.testGivenK(opListLossHistory[opIdx], currOpListLossHistory,
                                                     opListLossHistory[opIdx].size() - 1);
            if (maxProb < prob)
            {
                maxProb = prob;
                maxProbIdx = opIdx;
            }
        }
        foundOpModel = true;
        if (maxProbIdx != -1 and maxProb > pValueThreshold)
        {
            // this corresponds to removing the random MLP that was created
            destroyRandomNoTrainPettitt(agent);
            agent.currOp = maxProbIdx;
            std::deque<OpExperience> &opStateRef = opDequeStateHistory[agent.currOp];
            MLP &currOpRef = agent.opList[agent.currOp];
            // now train the currOp on the examples since they belong to it and add them to the history if maxHistory
            // was not achieved
            for (size_t idx = 0, sz = currOpListStateHistory.size(); idx != sz; ++idx)
            {
                OpExperience const &currExperience = currOpListStateHistory[idx];
                currOpRef.train(currExperience.lastState, currExperience.newState);
                if (opStateRef.size() < maxHistorySize)
                    opStateRef.push_back(currExperience);
                else
                {
                    opStateRef.pop_front();
                    opStateRef.push_back(currExperience);
                }
            }
        }
        updateCorrectPercentage(agent);
        return;
    }
    if (foundOpModel and opDequeStateHistory[agent.currOp].size() < maxHistorySize)
        opDequeStateHistory[agent.currOp].push_back({ lastState, newState });
    else if (foundOpModel and opDequeStateHistory[agent.currOp].size() >= maxHistorySize)
    {
        opDequeStateHistory[agent.currOp].pop_front();
        opDequeStateHistory[agent.currOp].push_back({ lastState, newState });
    }
}
void OpTrack::normalOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                               float loss)
{
}

void OpTrack::updateCorrectPercentage(Agent &agent)
{
    if (agent.currOp == agent.maze->getCurrSimState())
    {
        ++agent.correctOpCurrentEpisode;
    }
//    ++agent.opChoiceMatrix[agent.maze->getCurrSimState()][agent.currOp];
    ++agent.totalPredOpCurrentEpisode;
    agent.predictedOpponentType.push_back(agent.currOp);
    agent.actualOpponentType.push_back(agent.maze->getCurrSimState());

}
