// Pettitt's test for change-point detection with multiple versions including an adaptation as mentioned in the thesis
// Adapted from the trend package (https://cran.r-project.org/web/packages/trend/index.html) and translated into C++

// Copyright (c) 2016-2020, Thorsten Pohlert
// Copyright (c) 2021, Radu Alexandru Cosma

/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
