#include "sarsa.h"

Sarsa::Sarsa(size_t _nrEpisodes, float _alpha, float _epsilon)
    : Agent(_nrEpisodes), alpha(_alpha), epsilon(_epsilon)
{
}