#ifndef _INCLUDED_AGENT
#define _INCLUDED_AGENT

#include <cstddef>

#include "../simContainer/simContainer.h"
#include <vector>
#include "../simState/actions.h"
#include "../Eigen/Core"
#include "../mlp/mlp.h"
#include "../createRngObj/createRngObj.h"
#include "../monteCarloSim/monteCarloSim.h"
#include <cmath>

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
    ONEFORALL
};

class Agent
{
    protected:

    static constexpr size_t NR_ACTIONS = 4; // Hardcoded number of actions
    size_t d_killedByAshTime = 500;
    float Q_0 = 0;
    float runReward;
    size_t simTime = 0;
    SimContainer *maze;         // The maze the agent is navigating
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
    MLP mlp;
    MLP opponentMlp;
    bool isNewLevel = false;

    OpModellingType opModellingType;
    float gamma;
    size_t maxNrSteps = 10;
    size_t nrRollouts = 5;

  public:
    std::vector<float> const &getThisEpisodeLoss() const;

  public:
    std::vector<float> const &getOpponentCorrectPredictionPercentage() const;

  public:
    std::vector<float> const &getOpponentPredictionLosses() const;

  public:
    explicit Agent(size_t _nrEpisodes, OpModellingType pOpModellingType = OpModellingType::ONEFORALL, float pGamma = 0.99);
    virtual ~Agent();

    void run();

    std::vector<float> & getRewards();
    std::vector<size_t> & getHasDied();
    void setMaze(SimContainer *maze);
    float getRunReward();

    virtual bool performOneStep();
    virtual void newEpisode();
    virtual size_t actionWithQ(Eigen::VectorXf const &qVals);

    void handleOpponentAction();
    float MonteCarloRollout(size_t action);
    Eigen::VectorXf MonteCarloAllActions();
};


#endif
