#ifndef _INCLUDED_AGENT
#define _INCLUDED_AGENT

#include <cstddef>

#include "../../../Eigen/Core"
#include "../createRngObj/createRngObj.h"
#include "../kolsmir/kolsmir.h"
#include "../mlp/mlp.h"
#include "../monteCarloSim/monteCarloSim.h"
#include "../opTrack/opTrack.h"
#include "../pettitt/pettitt.h"
#include "../simContainer/simContainer.h"
#include "../simState/actions.h"
#include "../utilities/utilities.h"
#include "experience.h"
#include <cmath>
#include <vector>

enum class AgentType
{
    QLEARNING,
    DQLEARNING,
    SARSA,
    EXPECTEDSARSA
};

enum class OpModellingType
{
    NEWEVERYTIME,
    ONEFORALL,
    PETTITT,
    KOLSMIR,
    NOTRAINPETTITT
};

class Agent
{
    friend class OpTrack;

  protected:

    OpTrack opTrack;
    size_t const NR_ACTIONS = 4; // Hardcoded number of actions
    float runReward;
    size_t simTime = 0;
    SimContainer *maze; // The maze the agent is navigating
    bool opponentNotInit = true;
    Eigen::VectorXf lastState;
    Eigen::VectorXf lastOpponentState;
    size_t lastAction;
    size_t nrEpisodes;
    float currentEpisodeLoss;
    size_t currentEpisodeCorrectPredictions;
    std::vector<float> rewards;
    std::vector<size_t> hasDied;
    std::vector<float> opponentPredictionLosses;
    std::vector<float> opponentCorrectPredictionPercentage;
    std::vector<float> thisEpisodeLoss;

    size_t correctOpCurrentEpisode = 0;
    size_t totalPredOpCurrentEpisode = 0;

    MLP mlp;
    std::vector<MLP> opList;
    size_t currOp;

    bool isNewLevel = false;

    OpModellingType opModellingType;
    float alpha;
    float epsilon;
    float gamma;
    size_t maxNrSteps;
    size_t nrRollouts;
    std::vector<float> gammaVals;
    MLPParams opMLPParams;
    std::vector<size_t> opDeathsPerEp;

    void opPredict(void (OpTrack::*tracking)(Agent &agent, Eigen::VectorXf const &, Eigen::VectorXf const &, float));

  public:
    explicit Agent(OpTrackParams opTrackParams, AgentMonteCarloParams agentMonteCarloParams, MLPParams agentMLP,
                   MLPParams opponentMLP, size_t _nrEpisodes,
                   OpModellingType pOpModellingType, float pAlpha, float pEpsilon, float pGamma);
    virtual ~Agent();

    void run();

    std::vector<float> &getRewards();
    std::vector<size_t> &getHasDied();
    void setMaze(SimContainer *maze);
    float getRunReward();

    virtual bool performOneStep();
    virtual void newEpisode();
    virtual size_t actionWithQ(Eigen::VectorXf const &qVals);

    void handleOpponentAction();
    float MonteCarloRollout(size_t action);
    Eigen::VectorXf MonteCarloAllActions();
    std::vector<float> const &getThisEpisodeLoss() const;
    float getCorrectOpponentTypePredictionPercentage() const;
    std::vector<float> const &getOpponentCorrectPredictionPercentage() const;
    std::vector<float> const &getOpponentPredictionLosses() const;
    float getOpDeathPercentage() const;
};
inline float Agent::getCorrectOpponentTypePredictionPercentage() const
{
    return static_cast<float>(correctOpCurrentEpisode) / totalPredOpCurrentEpisode;
}
inline std::vector<float> const &Agent::getThisEpisodeLoss() const
{
    return thisEpisodeLoss;
}
inline std::vector<float> const &Agent::getOpponentCorrectPredictionPercentage() const
{
    return opponentCorrectPredictionPercentage;
}
inline std::vector<float> const &Agent::getOpponentPredictionLosses() const
{
    return opponentPredictionLosses;
}
inline float Agent::getRunReward()
{
    return runReward;
}
inline std::vector<size_t> &Agent::getHasDied()
{
    return hasDied;
}
inline std::vector<float> &Agent::getRewards()
{
    return rewards;
}
inline void Agent::setMaze(SimContainer *simCont)
{
    maze = simCont;
}


#endif
