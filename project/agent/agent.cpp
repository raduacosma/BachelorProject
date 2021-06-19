#include "agent.h"
#include "../simContainer/simContainer.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
// TODO: check out monte carlo estimates for target Q-values
using namespace std;
Agent::Agent(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
             MLPParams opponentMLP, size_t _nrEpisodes, size_t pNrEpisodesToEpsilonZero,
             OpModellingType pOpModellingType, float pEpsilon,
             float pGamma) // TODO: check how size is passed
    : opTrack(opTrackParams.pValueThreshold, opTrackParams.minHistorySize, opTrackParams.maxHistorySize),
      nrEpisodes(_nrEpisodes), nrEpisodesToEpsilonZero(pNrEpisodesToEpsilonZero), rewards(vector<float>(_nrEpisodes)),
      mlp(agentMLP.sizes, agentMLP.learningRate, agentMLP.regParam, agentMLP.outputActivationFunc,
          agentMLP.miniBatchSize, agentMLP.randInit),
      opList{ MLP(opponentMLP.sizes, opponentMLP.learningRate, opponentMLP.regParam, opponentMLP.outputActivationFunc,
                  opponentMLP.miniBatchSize, agentMLP.randInit) },
      opLosses{ 0.0f }, currOp(0), opModellingType(pOpModellingType), epsilon(pEpsilon), gamma(pGamma),
      maxNrSteps(agentMonteCarloParams.maxNrSteps), nrRollouts(agentMonteCarloParams.nrRollouts),
      opMLPParams(opponentMLP), opDeathsPerEp(nrEpisodes, 0)
{
    gammaVals.push_back(1); // in order to save some i-1s in monte carlo
    float tmpGamma = gamma;
    for (size_t idx = 0; idx <= maxNrSteps; ++idx) // although gammaVals[idx+1] is added
    {
        gammaVals.push_back(tmpGamma);
        tmpGamma *= gamma;
    }
    rewards.reserve(nrEpisodes);
    opponentPredictionLosses.reserve(nrEpisodes);
    opponentCorrectPredictionPercentage.reserve(nrEpisodes);
    thisEpisodeLoss.reserve(nrEpisodes);
    learningLosses.reserve(nrEpisodes);
}
bool Agent::performOneStep()
{
    //    throw std::runtime_error("In Agent's performOneStep, should not be here");
}
void Agent::run()
{
    // initialize Q values? They are set to 0 in stateSpaceSize and Marco's
    // slides say initialize "arbitrarily" while the book says terminal states
    // should be 0, so I guess initializing them all to 0 could be fine?

    // tracking stuff? avg rewards etc etc etc
    runReward = 0;
    float initialEpsilon = epsilon;
    float lastEpsilon = 0;
    float killedByOpponentReward = maze->getCurrentLevel().killedByOpponentReward();
    for (size_t nrEpisode = 0; nrEpisode != nrEpisodes; ++nrEpisode)
    {
        std::cout << epsilon << std::endl;
        // d_oldstate was modified from Maze so it's fine, anything else?
        newEpisode();
        currentEpisodeOpLoss = 0;
        currentEpisodeAgentLoss = 0;
        currentEpisodeCorrectPredictions = 0;
        size_t stepCount = 0;
        float totalReward = 0;
        initOpponentMethod();
        //        thisEpisodeLoss = std::vector<float>();  // this is for the old loss at switch stuff
        while (true)
        {
            ++stepCount;
            bool canContinue = performOneStep();
            float receivedReward = maze->getLastReward();
            totalReward += receivedReward;
            handleOpponentAction();
            if (not canContinue)
            {
                if (receivedReward == killedByOpponentReward)
                    opDeathsPerEp[nrEpisode] = 1;
                break;
            }
            if (stepCount >= 1000) // max nr of timesteps
            {
                maze->resetNextEpisode();
                break;
            }
            if (maze->getLastSwitchedLevel())
            {
                initOpponentMethod();
            }
        }
        std::cout << "totalReward: " << totalReward << std::endl;
        std::cout << opList.size() << std::endl;
        learningLosses.push_back(currentEpisodeAgentLoss / stepCount);
        //        if(nrEpisode%1000==0)
        //        {
        //            std::ofstream trainLoss{"results/trainLoss2.txt"};
        //            copy(learningLosses.begin(), learningLosses.end(),
        //                 std::ostream_iterator<float>(trainLoss, "\n"));
        //        }

        opponentPredictionLosses.push_back(currentEpisodeOpLoss / stepCount);
        opponentCorrectPredictionPercentage.push_back(static_cast<float>(currentEpisodeCorrectPredictions) / stepCount);
        runReward += totalReward;
        rewards[nrEpisode] = totalReward;
        if (epsilon > lastEpsilon)
            epsilon -= (initialEpsilon - lastEpsilon) / nrEpisodesToEpsilonZero;
    }
    // any cleanup?
}
void Agent::handleOpponentAction()
{
    switch (opModellingType)
    {
        case OpModellingType::NEWEVERYTIME:
        case OpModellingType::ONEFORALL:
            opPredict(&OpTrack::normalOpTracking);
            break;
        case OpModellingType::KOLSMIR:
            opPredictInterLoss(&OpTrack::kolsmirOpTracking); // TODO: handle level change in init
            break;
        case OpModellingType::BADLOSSPETTITT:
            opPredictInterLoss(&OpTrack::pettittOpTracking);
            break;
        case OpModellingType::NOTRAINPETTITT:
            opPredictInterLoss(&OpTrack::noTrainPettittOpTracking);
            break;
    }
}
void Agent::initOpponentMethod()
{ // same as a bit below? weird actually since it returns after might work
    // getStateForOpponent might be useless at this point
    // actually this is probably also fine since this is after a level switch I guess,
    // since opponentNotInit is set to true after the action is processed
    lastOpponentState = maze->getCurrentStateForOpponent();
    for (auto &item : opLosses)
        item = 0.0f;
//    if(not opTrack.isFoundOpModel())
//        currOp = opList.size() - 1;
    switch (opModellingType)
    {
        case OpModellingType::NEWEVERYTIME:
            opList[currOp].randomizeWeights();
            break;
        case OpModellingType::ONEFORALL:
            break;
        case OpModellingType::KOLSMIR:
            opTrack.kolsmirOpInit(*this);
            break;
        case OpModellingType::BADLOSSPETTITT:
            opTrack.pettittOpInit(*this);
            break;
        case OpModellingType::NOTRAINPETTITT:
            opTrack.noTrainPettittOpInit(*this);
            break;
    }
}
void Agent::opPredict(void (OpTrack::*tracking)(Agent &agent, Eigen::VectorXf const &, Eigen::VectorXf const &, float))
{ // TODO: check that level switching works properly
    Eigen::VectorXf newOpponentState = maze->getStateForOpponent();
    size_t newOpponentAction = maze->getLastOpponentAction();
    Eigen::VectorXf opponentActionTarget = Eigen::VectorXf::Zero(4);
    opponentActionTarget(static_cast<size_t>(newOpponentAction)) = 1.0f;
    size_t opponentActionIdx;
    opList[currOp].feedforward(lastOpponentState).maxCoeff(&opponentActionIdx);
    //    std::cout<<opponentActionIdx<<" "<<newOpponentAction<<std::endl;
    if (newOpponentAction == opponentActionIdx)
        ++currentEpisodeCorrectPredictions;
    float currentLoss = opList[currOp].update(opponentActionTarget);
    // TODO: Be careful with end of episode and reset and such
    currentEpisodeOpLoss += currentLoss;
    //    std::cout<<"Op loss: "<<currentLoss<<std::endl;
    (opTrack.*tracking)(*this, lastOpponentState, opponentActionTarget, currentLoss);
    //    thisEpisodeLoss.push_back(currentLoss); // TODO: only turn this on when needed
    lastOpponentState = newOpponentState;
}

void Agent::opPredictInterLoss(void (OpTrack::*tracking)(Agent &agent, Eigen::VectorXf const &, Eigen::VectorXf const &,
                                                         float))
{ // TODO: check that level switching works properly
    Eigen::VectorXf newOpponentState = maze->getStateForOpponent();
    size_t newOpponentAction = maze->getLastOpponentAction();
    Eigen::VectorXf opponentActionTarget = Eigen::VectorXf::Zero(4);
    opponentActionTarget(static_cast<size_t>(newOpponentAction)) = 1.0f;
    Eigen::VectorXf currPrediction = opList[currOp].predict(lastOpponentState);
    size_t opponentActionIdx;
    currPrediction.maxCoeff(&opponentActionIdx);
    //    std::cout<<opponentActionIdx<<" "<<newOpponentAction<<std::endl;
    if (newOpponentAction == opponentActionIdx)
        ++currentEpisodeCorrectPredictions;
    float currentLoss = opList[currOp].computeLoss(currPrediction, opponentActionTarget);
    // TODO: Be careful with end of episode and reset and such
    // this is also wrong but since the train loss metric overall is kind of useless for these methods meh
    currentEpisodeOpLoss += currentLoss;
    size_t currIdx = currOp;
    if (not opTrack.isFoundOpModel())
    {
        for (size_t idx = 0, sz = opList.size()-1; idx != sz; ++idx)
            opLosses[idx] += opList[idx].predictWithLoss(lastOpponentState, opponentActionTarget);
        int minIdx = -1;
        float minLoss = std::numeric_limits<float>::max();
        for (int idx = 0, sz = opLosses.size()-1; idx != sz; ++idx)
            if (minLoss > opLosses[idx])
            {
                minIdx = idx;
                minLoss = opLosses[idx];
            }
        if (minIdx >= 0)
            currIdx = minIdx;
        currOp = opList.size() - 1;
    }
    //    std::cout<<"Op loss: "<<currentLoss<<std::endl;
    opList[currOp].train(lastOpponentState, opponentActionTarget);
    // currentLoss here is bad since it should be the train loss on currOp, but since that is used for old pettitt
    // I'll leave it like this
    (opTrack.*tracking)(*this, lastOpponentState, opponentActionTarget, currentLoss);
    //    thisEpisodeLoss.push_back(currentLoss); // TODO: only turn this on when needed
    if (not opTrack.isFoundOpModel())
        currOp = currIdx;
    lastOpponentState = newOpponentState;
}
// if I pass an agentState here it should already be the current state in maze that I will get anyways
float Agent::MonteCarloRollout(size_t action)
{
    auto &rngEngine = globalRng.getRngEngine();
    float totalReward = 0;
    for (size_t mIdx = 0; mIdx != nrRollouts; ++mIdx)
    {
        float rolloutReward = 0;
        size_t i = 0;
        MonteCarloSim copyMaze(maze->getCurrentLevel());
        Eigen::VectorXf opProbs = opList[currOp].predict(copyMaze.getStateForOpponent());
        size_t opAction;
        if (nrRollouts == 1)
        {
            opProbs.maxCoeff(&opAction);
        }
        else
        {
            std::discrete_distribution<> distr({ opProbs[0], opProbs[1], opProbs[2], opProbs[3] });
            opAction = distr(rngEngine);
        }

        // always call updateOpPos before computeNextState and getting the state
        auto [reward, canContinue] =
            copyMaze.computeNextStateAndReward(static_cast<Actions>(action), static_cast<Actions>(opAction));
        rolloutReward += reward;
        if (canContinue != SimResult::KILLED_BY_OPPONENT and canContinue != SimResult::REACHED_GOAL and maxNrSteps > 0)
        {
            while (true)
            {
                ++i;

                Eigen::VectorXf innerOpProbs = opList[currOp].predict(copyMaze.getStateForOpponent());
                size_t innerOpAction;
                if (nrRollouts == 1)
                {
                    innerOpProbs.maxCoeff(&innerOpAction);
                }
                else
                {
                    std::discrete_distribution<> innerDistr(
                        { innerOpProbs[0], innerOpProbs[1], innerOpProbs[2], innerOpProbs[3] });
                    innerOpAction = innerDistr(rngEngine);
                }
                size_t agentAction;
                mlp.predict(copyMaze.getStateForAgent())
                    .maxCoeff(&agentAction); // this and last if should have the same agent state one call to the other
                auto [innerReward, innerCanContinue] = copyMaze.computeNextStateAndReward(
                    static_cast<Actions>(agentAction), static_cast<Actions>(innerOpAction));
                rolloutReward += gammaVals[i] * innerReward;
                if (innerCanContinue == SimResult::KILLED_BY_OPPONENT or innerCanContinue == SimResult::REACHED_GOAL)
                {
                    break;
                }
                if (i == maxNrSteps)
                {
                    rolloutReward += gammaVals[i] * mlp.predict(copyMaze.getStateForAgent()).maxCoeff();
                    break;
                }
            }
        }
        else if (canContinue == SimResult::CONTINUE and
                 maxNrSteps == 0) // 1 or 0 here? since maxNrSteps corresponds to the while
        {
            rolloutReward += mlp.predict(copyMaze.getStateForAgent()).maxCoeff();
        }
        totalReward += rolloutReward;
    }
    return totalReward / nrRollouts;
}

size_t Agent::actionWithQ(Eigen::VectorXf const &qVals) const
{
    bool explore = globalRng.getUniReal01() < epsilon;
    size_t choice;
    if (explore)
    {
        choice = globalRng.getUniReal01() * NR_ACTIONS;
    }
    else
    {
        float maxVal = qVals.maxCoeff();
        std::vector<size_t> maxIdxs;
        maxIdxs.reserve(4);
        for (size_t idx = 0; idx != NR_ACTIONS; ++idx)
            if (qVals[idx] == maxVal)
                maxIdxs.push_back(idx);
        choice = maxIdxs[globalRng.getUniReal01() * maxIdxs.size()];
        //        qVals.maxCoeff(&choice);
    }

    return choice;
}
Eigen::VectorXf Agent::MonteCarloAllActions()
{
    Eigen::VectorXf estimatedQValues(4);
    for (size_t idx = 0; idx != NR_ACTIONS; ++idx)
        estimatedQValues[idx] = MonteCarloRollout(idx);
    return estimatedQValues;
}
void Agent::newEpisode()
{
    lastState = maze->getStateForAgent();
    lastOpponentState = maze->getStateForOpponent();
    // Some algorithms require this. Empty for the others.
}

Agent::~Agent()
{
}

float Agent::getOpDeathPercentage() const
{
    size_t count = 0;
    for (auto const &item : opDeathsPerEp)
        if (item == 1)
            ++count;
    return static_cast<float>(count) / nrEpisodes;
}
