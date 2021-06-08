#include "agent.h"
#include "../simContainer/simContainer.h"
#include <iostream>
#include <random>
#include <tuple>
// TODO: check out monte carlo estimates for target Q-values
using namespace std;
Agent::Agent(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
             MLPParams opponentMLP, size_t _nrEpisodes, OpModellingType pOpModellingType,
             float pAlpha, float pEpsilon, float pGamma) // TODO: check how size is passed
    : opTrack(opTrackParams.pValueThreshold, opTrackParams.minHistorySize, opTrackParams.maxHistorySize),
      nrEpisodes(_nrEpisodes), rewards(vector<float>(_nrEpisodes)),
      mlp(agentMLP.sizes, agentMLP.learningRate, agentMLP.outputActivationFunc, agentMLP.miniBatchSize),
      opList{ MLP(opponentMLP.sizes, opponentMLP.learningRate, opponentMLP.outputActivationFunc,
                  opponentMLP.miniBatchSize) },
      currOp(0), opModellingType(pOpModellingType), alpha(pAlpha), epsilon(pEpsilon),gamma(pGamma), maxNrSteps(agentMonteCarloParams.maxNrSteps),
      nrRollouts(agentMonteCarloParams.nrRollouts), opMLPParams(opponentMLP),opDeathsPerEp(nrEpisodes,0)
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
}
bool Agent::performOneStep()
{
    throw std::runtime_error("In Agent's performOneStep, should not be here");
}
void Agent::run()
{
    // initialize Q values? They are set to 0 in stateSpaceSize and Marco's
    // slides say initialize "arbitrarily" while the book says terminal states
    // should be 0, so I guess initializing them all to 0 could be fine?

    // tracking stuff? avg rewards etc etc etc
    runReward = 0;
    float killedByOpponentReward = maze->getCurrentLevel().killedByOpponentReward();
    for (size_t nrEpisode = 0; nrEpisode != nrEpisodes; ++nrEpisode)
    {
        // d_oldstate was modified from Maze so it's fine, anything else?
        newEpisode();
        currentEpisodeLoss = 0;
        currentEpisodeCorrectPredictions = 0;
        size_t stepCount = 0;
        float totalReward = 0;
        opponentNotInit = true;
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
                if(receivedReward == killedByOpponentReward)
                    opDeathsPerEp[nrEpisode] = 1;
                break;
            }
            if (maze->getLastSwitchedLevel())
            {
                opponentNotInit = true;
            }
        }
        std::cout << "totalReward: " << totalReward << std::endl;
        std::cout<< opList.size()<<std::endl;
        opponentPredictionLosses.push_back(currentEpisodeLoss / stepCount);
        opponentCorrectPredictionPercentage.push_back(static_cast<float>(currentEpisodeCorrectPredictions) / stepCount);
        runReward += totalReward;
        rewards[nrEpisode] = totalReward;
    }
    // any cleanup?
}
void Agent::handleOpponentAction()
{
    if (opponentNotInit)
    {
        // same as a bit below? weird actually since it returns after might work
        // getStateForOpponent might be useless at this point
        // actually this is probably also fine since this is after a level switch I guess,
        // since opponentNotInit is set to true after the action is processed
        lastOpponentState = maze->getStateForOpponent();
        opponentNotInit = false;
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
            case OpModellingType::PETTITT:
                opTrack.pettittOpInit(*this);
                break;
            case OpModellingType::NOTRAINPETTITT:
                opTrack.noTrainPettittOpInit(*this);
                break;
        }

        return;
    }
    switch (opModellingType)
    {
        case OpModellingType::NEWEVERYTIME:
        case OpModellingType::ONEFORALL:
            opPredict(&OpTrack::normalOpTracking);
            break;
        case OpModellingType::KOLSMIR:
            opPredict(&OpTrack::kolsmirOpTracking); // TODO: handle level change in init
            break;
        case OpModellingType::PETTITT:
            opPredict(&OpTrack::pettittOpTracking);
            break;
        case OpModellingType::NOTRAINPETTITT:
            opPredict(&OpTrack::noTrainPettittOpTracking);
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
    currentEpisodeLoss += currentLoss;
    (opTrack.*tracking)(*this, lastOpponentState, opponentActionTarget, currentLoss);
    //    thisEpisodeLoss.push_back(currentLoss); // TODO: only turn this on when needed
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
        std::discrete_distribution<> distr({ opProbs[0], opProbs[1], opProbs[2], opProbs[3] });
        size_t opAction = distr(rngEngine);

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
                std::discrete_distribution<> innerDistr(
                    { innerOpProbs[0], innerOpProbs[1], innerOpProbs[2], innerOpProbs[3] });
                size_t innerOpAction = innerDistr(rngEngine);
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
size_t Agent::actionWithQ(Eigen::VectorXf const &qVals)
{
    throw std::runtime_error("In Agent's actionWithQ, should not be here");
}



float Agent::getOpDeathPercentage() const
{
    size_t count = 0;
    for(auto const &item:opDeathsPerEp)
        if(item == 1)
            ++count;
    return static_cast<float>(count)/nrEpisodes;
}
