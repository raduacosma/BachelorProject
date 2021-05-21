#ifndef _INCLUDED_PETTITT
#define _INCLUDED_PETTITT
#include <vector>
#include <tuple>
class Pettitt
{
  public:
    std::tuple<double,double,int> test(std::vector<double> const &x);
  private:
    std::vector<double> ranks(std::vector<double> const &x);
};
#endif
