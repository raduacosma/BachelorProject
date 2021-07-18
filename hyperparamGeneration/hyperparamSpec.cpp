#include "hyperparamSpec.h"

std::ostream &operator<<(std::ostream &out, OpModellingType const &opModellingType)
{
    out << static_cast<int>(opModellingType);
    return out;
}
std::istream &operator>>(std::istream &in, OpModellingType &opModellingType)
{
    int enumVal;
    in >> enumVal;
    opModellingType = static_cast<OpModellingType>(enumVal);
    return in;
}
std::ostream &operator<<(std::ostream &out, AgentType const &agentType)
{
    out << static_cast<int>(agentType);
    return out;
}
std::istream &operator>>(std::istream &in, AgentType &agentType)
{
    int enumVal;
    in >> enumVal;
    agentType = static_cast<AgentType>(enumVal);
    return in;
}
std::ostream &operator<<(std::ostream &out, HyperparamSpec const &hs)
{
    out << "seed\n"
        << hs.seed << '\n'
        << "files\n"
        << hs.files << '\n'
        << "miniBatchSize\n"
        << hs.miniBatchSize << '\n'
        << "numberOfEpisodes\n"
        << hs.numberOfEpisodes << '\n'
        << "sizeExperience\n"
        << hs.sizeExperience << '\n'
        << "agentType\n"
        << hs.agentType << '\n'
        << "epsilon\n"
        << hs.epsilon << '\n'
        << "gamma\n"
        << hs.gamma << '\n'
        << "agentVisionGridSize\n"
        << hs.agentVisionGridSize << '\n'
        << "opponentVisionGridSize\n"
        << hs.opponentVisionGridSize << '\n'
        << "swapPeriod\n"
        << hs.swapPeriod << '\n'
        << "maxNrSteps\n"
        << hs.maxNrSteps << '\n'
        << "nrRollouts\n"
        << hs.nrRollouts << '\n'
        << "agentLearningRate\n"
        << hs.agentLearningRate << '\n'
        << "agentRegParam\n"
        << hs.agentRegParam << '\n'
        << "opponentLearningRate\n"
        << hs.opponentLearningRate << '\n'
        << "opponentRegParam\n"
        << hs.opponentRegParam << '\n'
        << "traceSize\n"
        << hs.traceSize << '\n'
        << "randomOpCoef\n"
        << hs.randomOpCoef << '\n'
        << "opModellingType\n"
        << hs.opModellingType << '\n'
        << "pValueThreshold\n"
        << hs.pValueThreshold << '\n'
        << "minHistorySize\n"
        << hs.minHistorySize << '\n'
        << "maxHistorySize\n"
        << hs.maxHistorySize << '\n';
    return out;
}

std::istream &operator>>(std::istream &in, HyperparamSpec &hs)
{
    writeField(in, "seed", hs.seed);
    writeField(in, "files", hs.files);
    writeField(in, "miniBatchSize", hs.miniBatchSize);
    writeField(in, "numberOfEpisodes", hs.numberOfEpisodes);
    writeField(in, "sizeExperience", hs.sizeExperience);
    writeField(in, "agentType", hs.agentType);
    writeField(in, "epsilon", hs.epsilon);
    writeField(in, "gamma", hs.gamma);
    writeField(in, "agentVisionGridSize", hs.agentVisionGridSize);
    writeField(in, "opponentVisionGridSize", hs.opponentVisionGridSize);
    writeField(in, "swapPeriod", hs.swapPeriod);
    writeField(in, "maxNrSteps", hs.maxNrSteps);
    writeField(in, "nrRollouts", hs.nrRollouts);
    writeField(in, "agentLearningRate", hs.agentLearningRate);
    writeField(in, "agentRegParam", hs.agentRegParam);
    writeField(in, "opponentLearningRate", hs.opponentLearningRate);
    writeField(in, "opponentRegParam", hs.opponentRegParam);
    writeField(in, "traceSize", hs.traceSize);
    writeField(in, "randomOpCoef", hs.randomOpCoef);
    writeField(in, "opModellingType", hs.opModellingType);
    writeField(in, "pValueThreshold", hs.pValueThreshold);
    writeField(in, "minHistorySize", hs.minHistorySize);
    writeField(in, "maxHistorySize", hs.maxHistorySize);
    return in;
}