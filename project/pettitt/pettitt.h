#ifndef _INCLUDED_PETTITT
#define _INCLUDED_PETTITT
#include <vector>
#include <tuple>
class Pettitt
{
  public:
    std::tuple<double,double,int> test(std::vector<double> const &x);
    std::tuple<double,double,int> test2(std::vector<double> const &x, std::vector<double> const &x2);
  private:
    std::vector<double> ranks(std::vector<double> const &x);
    std::vector<double> ranks2(std::vector<double> const &x, std::vector<double> const &x2);
};
#endif