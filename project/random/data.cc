#include "random.ih"

std::normal_distribution<double> Random::s_normal { 0, 1 };
std::uniform_real_distribution<double> Random::s_uniform { 0, 1 };
std::default_random_engine Random::s_engine { 1 };        // Seed for reproducability
