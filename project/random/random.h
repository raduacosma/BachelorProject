#ifndef INCLUDED_RANDOM_
#define INCLUDED_RANDOM_

#include <random>
#include <cstddef>

class Random
{
  static std::uniform_real_distribution<double> s_uniform;
  static std::normal_distribution<double> s_normal;
  static std::default_random_engine s_engine;

  public:
    static double normal();
    static double uniform();
    static size_t uniformInt(size_t start, size_t end);
    static void seed(size_t seed);
};

inline double Random::normal()
{
  return s_normal(s_engine);
}

inline double Random::uniform()
{
  return s_uniform(s_engine);
}
inline void Random::seed(size_t seed)
{
  s_engine = std::default_random_engine { seed };
}
        
#endif
