#ifndef _INCLUDED_PETTITT
#define _INCLUDED_PETTITT
#include <deque>
#include <tuple>
#include <vector>

class Pettitt
{
  public:
    std::tuple<double, double, int> test(std::vector<double> const &x);
    std::tuple<double, double, int> test2(std::deque<double> const &x, std::vector<double> const &x2);
    std::tuple<double, double, int> test2(std::vector<double> const &x, std::vector<double> const &x2);
    std::tuple<double, double, int> testGivenK(std::deque<double> const &x, std::vector<double> const &x2, int K);
    std::tuple<double, double, int> testGivenK(std::vector<double> const &x, std::vector<double> const &x2, int K);

  private:
    std::vector<double> ranks(std::vector<double> const &x);
    std::vector<double> ranks2(std::deque<double> const &x, std::vector<double> const &x2);
    std::vector<double> ranks2(std::vector<double> const &x, std::vector<double> const &x2);
};
#endif
