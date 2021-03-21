#include "random.ih"

size_t Random::uniformInt(size_t start, size_t end)
{
  uniform_int_distribution<int> distr(start,end);
  return distr(s_engine);
}