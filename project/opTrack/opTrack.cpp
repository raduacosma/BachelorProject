#include "opTrack.h"
#include "../agent/agent.h"
#include "../kolsmir/kolsmir.h"
#include "../pettitt/pettitt.h"

void OpTrack::destroyRandomKolsmir(Agent &agent)
{
    destroyRandomPettitt(agent);
    opListStateHistory.pop_back();
}
void OpTrack::destroyRandomPettitt(Agent &agent)
{
    agent.opList.pop_back();
    opListLossHistory.pop_back();
}
void OpTrack::commonOpInit(Agent &agent)
{
    opHistoryCounter = 0;
    foundOpModel = false;
    currOpListLossHistory.resize(0);
    // might need to pre-allocate these since otherwise there will be size problems
    currOpListStateHistory.resize(0);
    opListLossHistory.emplace_back();
    opListLossHistory.back().reserve(maxHistorySize);
    if (firstTime)
    {
        firstTime = false;
        currOpListLossHistory.reserve(minHistorySize);
        currOpListStateHistory.reserve(minHistorySize);
        return;
    }
    // probably need an initializer list or something so it jives with the initial opponentMlp hyperparams
    agent.opList.push_back(MLP({ 50, 100, 4 }, 0.001, ActivationFunction::SOFTMAX));
    agent.currOp = agent.opList.size() - 1;
}
// TODO: I need to also dynamically manage opListStateHistory and Losses
void OpTrack::kolsmirOpInit(Agent &agent)
{

    // need to account for the first time this is run with a random opponent
    // this will probably always be the random one so I could just to pop_back()
    if (not firstTime and opListStateHistory[agent.currOp].size() < minHistorySize)
    {
        destroyRandomKolsmir(agent);
    }
    opListStateHistory.emplace_back();
    opListStateHistory.back().reserve(maxHistorySize);
    commonOpInit(agent);
}
void OpTrack::pettittOpInit(Agent &agent)
{

    // need to account the first time this is run with a random opponent
    // this will probably always be the random one so I could just to pop_back()
    if (not firstTime and opListLossHistory[agent.currOp].size() < minHistorySize)
    {
        destroyRandomPettitt(agent);
    }
    commonOpInit(agent);
}
void OpTrack::pettittOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                                float loss)
{
    if (not foundOpModel and opHistoryCounter < minHistorySize)
    {
        currOpListStateHistory.push_back({ lastState, newState });
        ++opHistoryCounter;
    }
    else if (not foundOpModel and opHistoryCounter > minHistorySize) // TODO: check this condition
    {
        // TODO: check that all the vectors and indices and sizes are what they should be
        double maxProb = -1;
        int maxProbIdx = -1;
        std::vector<double> maxOpListLossHistory;
        // -1 for below since we don't want the random one that will be put last
        for (size_t opIdx = 0, opSz = opCopies.size() - 1; opIdx != opSz; ++opIdx)
        {
            MLP &currDoneOp = opCopies[opIdx];
            currOpListLossHistory.resize(0);
            std::transform(currOpListStateHistory.cbegin(), currOpListStateHistory.cend(),
                           std::back_inserter(currOpListLossHistory),
                           [&](OpExperience const &currExperience)
                           {
                               return currDoneOp.train(currExperience.lastState, currExperience.newState);
                           });
            auto [prob, U, K] = Pettitt{}.test2(opListLossHistory[opIdx], currOpListLossHistory);
            if (maxProb < prob)
            {
                maxProb = prob;
                maxProbIdx = opIdx;
                maxOpListLossHistory = currOpListLossHistory;
            }
        }
        // a model will be found anyways, whether random or a previous one, this is not the thing
        // that checks whether we are using a previous MLP, but that check is probably useless if
        // we don't want to avoid the minimum if
        foundOpModel = true;
        // opHistoryCounter is already done in init
        if (maxProbIdx != -1 and maxProb > pValueThreshold)
        {
            // this should correspond to removing the random MLP that was created
            destroyRandomPettitt(agent); // TODO: check that the pop_backs of curr state history is ok
            agent.currOp = maxProbIdx;
            std::vector<double> &opLossRef = opListLossHistory[agent.currOp];
            // now train the currOp on the examples since they belong to it and add them to the history if maxHistory
            // was not achieved
            for (size_t idx = 0, sz = maxOpListLossHistory.size(); idx != sz; ++idx)
            {
                if (opLossRef.size() < maxHistorySize)
                    opLossRef.push_back(maxOpListLossHistory[idx]);
            }
        }
        // nothing left to do since the random MLP was already here and we don't need to remove it

    }
    else if (foundOpModel and opListLossHistory[agent.currOp].size() < maxHistorySize)
        opListLossHistory[agent.currOp].push_back(loss); // if there are problems, check this loss thing
}
// TODO: don't forget to sort for kolsmir
// TODO: check that the logic from begin and end is properly split between tracking and init for all such member
// functions
void OpTrack::kolsmirOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                                float loss)
{
    if (not foundOpModel and opHistoryCounter < minHistorySize)
    {
        currOpListStateHistory.push_back({ lastState, newState });
        ++opHistoryCounter;
    }
    else if (not foundOpModel and opHistoryCounter > minHistorySize) // TODO: check this condition
    {
        // TODO: check that all the vectors and indices and sizes are what they should be
        double maxProb = -1;
        int maxProbIdx = -1; // -1 for below since we don't want the random one that will be put last
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

            auto [prob, rdmax] = KolSmir{}.test(opListLossHistory[opIdx].size(), opListLossHistory[opIdx].data(),
                                                currOpListLossHistory.size(), currOpListLossHistory.data());
            if (maxProb < prob)
            {
                maxProb = prob;
                maxProbIdx = opIdx;
            }
        }
        // a model will be found anyways, whether random or a previous one, this is not the thing
        // that checks whether we are using a previous MLP, but that check is probably useless if
        // we don't want to avoid the minimum if
        foundOpModel = true;
        if (maxProbIdx != -1 and maxProb > pValueThreshold)
        {
            // this should correspond to removing the random MLP that was created
            destroyRandomKolsmir(agent); // TODO: check that the pop_backs of curr state history is ok
            agent.currOp = maxProbIdx;
            std::vector<OpExperience> &opStateRef = opListStateHistory[agent.currOp];
            std::vector<double> &opLossRef = opListLossHistory[agent.currOp];
            MLP &currOpRef = agent.opList[agent.currOp];
            // now train the currOp on the examples since they belong to it and add them to the history if maxHistory
            // was not achieved
            for (size_t idx = 0, sz = currOpListStateHistory.size(); idx != sz; ++idx)
            {
                OpExperience const &currExperience = currOpListStateHistory[idx];
                currOpRef.train(currExperience.lastState, currExperience.newState);
                if (opStateRef.size() < maxHistorySize)
                    opStateRef.push_back({ currExperience.lastState, currExperience.newState });
            }
            // and update the losses for the last examples
            opLossRef.resize(0);
            std::transform(opStateRef.cbegin(), opStateRef.cend(), std::back_inserter(opLossRef),
                           [&](OpExperience const &currExperience)
                           {
                               return currOpRef.predictWithLoss(currExperience.lastState, currExperience.newState);
                           });
        }
        // nothing left to do since the random MLP was already here and we don't need to remove it

    }
    else if (foundOpModel and opListStateHistory[agent.currOp].size() < maxHistorySize)
        opListStateHistory[agent.currOp].push_back({ lastState, newState });
}

void OpTrack::normalOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                               float loss)
{
}