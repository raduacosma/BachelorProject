#ifndef _INCLUDED_OPTYPES
#define _INCLUDED_OPTYPES

enum class OpModellingType
{
    NEWEVERYTIME,
    ONEFORALL,
    KOLSMIR,
    NOTRAINPETTITT,
    BADLOSSPETTITT
};
enum class AgentType
{
    SARSA,
    DEEPQLEARNING,
    DOUBLEDEEPQLEARNING
};
#endif
