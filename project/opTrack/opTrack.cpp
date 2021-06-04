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
    opListLossHistory.pop_back();
    opListStateHistory.pop_back();
}
void OpTrack::destroyRandomPettitt(Agent &agent)
{
    agent.opList.pop_back();
    opDequeLossHistory.pop_back();
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
        currOpListLossHistory.reserve(maxHistorySize); // this is called after resize above, problem?
        currOpListStateHistory.reserve(maxHistorySize);
        return;
    }
    // this is just because the other baseline algorithms only use 1 MLP and therefore the initial opList
    // gets initialised with one MLP which we need to take care of the first time
    // probably need an initializer list or something so it jives with the initial opponentMlp hyperparams
    agent.opList.emplace_back(agent.opMLPParams.sizes, agent.opMLPParams.learningRate,
                              agent.opMLPParams.outputActivationFunc, agent.opMLPParams.miniBatchSize);
    agent.currOp = agent.opList.size() - 1;
}

void OpTrack::kolsmirOpInit(Agent &agent)
{

    // need to account for the first time this is run with a random opponent
    // this will probably always be the random one so I could just to pop_back()
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

    // need to account the first time this is run with a random opponent
    // this will probably always be the random one so I could just to pop_back()
    if (not firstTime and (opDequeLossHistory[agent.currOp].size() < minHistorySize or not foundOpModel))
    {
//        std::cout<<"IN IF"<<std::endl;
        destroyRandomPettitt(agent);
    }

//    std::cout<<"OUTSIDE IF"<<std::endl;
    if (firstTime)
        opCopies = std::vector<MLP>();
    else
        opCopies = agent.opList;
    opDequeLossHistory.emplace_back();
    // not possible with deque and also only done for performance reasons
    //    opDequeLossHistory.back().reserve(maxHistorySize);
    commonOpInit(agent);
}
void OpTrack::pettittOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                                float loss)
{
    if (not foundOpModel and opHistoryCounter < minHistorySize)
    {
//        std::cout<<"before;"<<std::endl;
        currOpListStateHistory.push_back({ lastState, newState });
        // we need to place the current losses for the random opponent
        // so do it here
        // no checks needed since this will be exactly minHistorySize
        opDequeLossHistory[agent.currOp].push_back(loss); // if there are problems, check this loss thing
        ++opHistoryCounter;
        return;
    }
    if (not foundOpModel and opHistoryCounter >= minHistorySize) // TODO: check this condition
    {
//        std::cout<<"Hello"<<std::endl;
        // TODO: check that all the vectors and indices and sizes are what they should be
        double maxProb = -1;
        int maxProbIdx = -1;
        std::vector<double> maxOpListLossHistory;

        double maxChangeAfterProb = -1;
        int maxChangeAfterProbIdx = -1;
        std::vector<double> maxChangeAfterOpListLossHistory;
        if (!opCopies.empty())
        {
//            std::cout<<"I AM IN"<<std::endl;
            assert(opCopies.size()==agent.opList.size()-1);
            // copy is done before random is added, for firstTime care is taken that the random is not added
            for (size_t opIdx = 0, opSz = opCopies.size(); opIdx != opSz; ++opIdx)
            {
                MLP &currDoneOp = opCopies[opIdx];
                currOpListLossHistory.resize(0);
                for(auto const &currState:currOpListStateHistory)
                {
                    currOpListLossHistory.push_back(currDoneOp.train(currState.lastState,currState.newState));
                }
                // TODO: wtf happens when  there is nothing in the loss history?
                // though if there is nothing in history then it should not even be here since only the ones
                // with minHistory should be in history
                auto [prob, U, K] = Pettitt{}.test2(opDequeLossHistory[opIdx], currOpListLossHistory);
//                                std::cout<<"pettitt: "<<prob<<" "<<U<<" "<<K<<std::endl;
                if (maxProb < prob)
                {
                    maxProb = prob;
                    maxProbIdx = opIdx;
                    maxOpListLossHistory = currOpListLossHistory;
                }
                if (K + 1 < opDequeLossHistory[opIdx].size() and maxChangeAfterProb < prob and prob < pValueThreshold)
                {
                    //                    std::cout<<"yesssss"<<std::endl;
                    maxChangeAfterProb = prob;
                    maxChangeAfterProbIdx = opIdx;
                    maxChangeAfterOpListLossHistory = currOpListLossHistory;
                }
//                                std::cout<<"begin loss history"<<std::endl;
//                                for (auto const &item : opDequeLossHistory[opIdx])
//                                {
//                                    std::cout << item << ',';
//                                }
//                                for (auto const &item : currOpListLossHistory)
//                                {
//                                    std::cout << item << ',';
//                                }
//                                std::cout<<std::endl;
//                                std::cout<<"end loss history"<<std::endl;
            }
        }
        // a model will be found anyways, whether random or a previous one, this is not the thing
        // that checks whether we are using a previous MLP, but that check is probably useless if
        // we don't want to avoid the minimum if
        foundOpModel = true;
        // opHistoryCounter is already done in init
        bool opModelChanged = false;
        if (maxProbIdx != -1 and maxProb > pValueThreshold)
        {
            opModelChanged = true;
        }
        else if (maxChangeAfterProbIdx != -1)
        {
            maxProbIdx = maxChangeAfterProbIdx;
            maxOpListLossHistory = maxChangeAfterOpListLossHistory;
            opModelChanged = true;
        }
        if (opModelChanged)
        {
            // this should correspond to removing the random MLP that was created
            destroyRandomPettitt(agent); // TODO: check that the pop_backs of curr state history is ok
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

//                std::cout<<"begin: "<<std::endl;
//                std::cout<<"p value: "<<maxProb<<std::endl;
//                std::cout<<agent.currOp<<" "<<agent.maze->getCurrSimState()<<std::endl;
//                std::cout<<"end: "<<std::endl;
        // nothing to do here since this is the random opponent and it should already have the state
        // incremented above
        updateCorrectPercentage(agent);
    }
    if (foundOpModel and opDequeLossHistory[agent.currOp].size() < maxHistorySize)
        opDequeLossHistory[agent.currOp].push_back(loss); // if there are problems, check this loss thing
    else if (foundOpModel and opDequeLossHistory[agent.currOp].size() >= maxHistorySize)
    {
        opDequeLossHistory[agent.currOp].pop_front();
        opDequeLossHistory[agent.currOp].push_back(loss); // if there are problems, check this loss thing
    }
}
// TODO: check that the logic from begin and end is properly split between tracking and init for all such member
// functions
void OpTrack::kolsmirOpTracking(Agent &agent, Eigen::VectorXf const &lastState, Eigen::VectorXf const &newState,
                                float loss)
{
    if (not foundOpModel and opHistoryCounter < minHistorySize)
    {
        // for kolsmir, currOpListStateHistory can be the same as the state history for the random one, as that is what
        // is done here. However, in order for it to be similar with pettitt, we leave it like this for now
        currOpListStateHistory.push_back({ lastState, newState });
        //        opListStateHistory[agent.currOp].push_back({ lastState, newState });
        ++opHistoryCounter;
        return;
    }
    if (not foundOpModel and opHistoryCounter >= minHistorySize) // TODO: check this condition
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
        // nothing to do here since this is the random opponent and it should already have the state
        // incremented above
    }
    // this should get triggered even if the above if is entered, same for pettitt
    if (foundOpModel and opListStateHistory[agent.currOp].size() < maxHistorySize)
        opListStateHistory[agent.currOp].push_back({ lastState, newState });
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
    ++agent.totalPredOpCurrentEpisode;
}