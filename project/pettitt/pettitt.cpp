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

#include "pettitt.h"
#include "../../Eigen/Core"
#include <iostream>
#include <numeric>

std::tuple<double, double, int> Pettitt::testGivenK(std::vector<double> const &x, std::vector<double> const &x2, int K)
{
    size_t n = x.size() + x2.size();
    Eigen::ArrayXd k(n);
    for (size_t idx = 0; idx != n; ++idx)
        k(idx) = idx + 1;

    std::vector<double> rank = ranks2(x, x2);
    std::vector<double> rankPartialSum;
    rankPartialSum.reserve(n);
    rankPartialSum.push_back(rank[0]);
    for (size_t idx = 1; idx != n; ++idx)
    {
        rankPartialSum.push_back(rank[idx] + rankPartialSum.back());
    }
    auto func = [&](double x)
    {
        return 2 * rankPartialSum[x - 1] - x * (n + 1);
    };
    Eigen::ArrayXd Uk = k.unaryExpr(func);
    // K will take into account 0 numbering
    double U = std::abs(Uk[K]);
    double pval = std::min(1.0, 2 * std::exp((-6 * U * U) / (n * n * (n + 1))));
    return std::make_tuple(pval, U, K);
}
std::tuple<double, double, int> Pettitt::testGivenK(std::deque<double> const &x, std::vector<double> const &x2, int K)
{
    size_t n = x.size() + x2.size();
    Eigen::ArrayXd k(n);
    for (size_t idx = 0; idx != n; ++idx)
        k(idx) = idx + 1;

    std::vector<double> rank = ranks2(x, x2);
    std::vector<double> rankPartialSum;
    rankPartialSum.reserve(n);
    rankPartialSum.push_back(rank[0]);
    for (size_t idx = 1; idx != n; ++idx)
    {
        rankPartialSum.push_back(rank[idx] + rankPartialSum.back());
    }
    auto func = [&](double x)
    {
        return 2 * rankPartialSum[x - 1] - x * (n + 1);
    };
    Eigen::ArrayXd Uk = k.unaryExpr(func);
    // K will take into account 0 numbering
    double U = std::abs(Uk[K]);
    double pval = std::min(1.0, 2 * std::exp((-6 * U * U) / (n * n * (n + 1))));
    return std::make_tuple(pval, U, K);
}
std::tuple<double, double, int> Pettitt::test2(std::vector<double> const &x, std::vector<double> const &x2)
{
    size_t n = x.size() + x2.size();
    Eigen::ArrayXd k(n);
    for (size_t idx = 0; idx != n; ++idx)
        k(idx) = idx + 1;

    std::vector<double> rank = ranks2(x, x2);
    std::vector<double> rankPartialSum;
    rankPartialSum.reserve(n);
    rankPartialSum.push_back(rank[0]);
    for (size_t idx = 1; idx != n; ++idx)
    {
        rankPartialSum.push_back(rank[idx] + rankPartialSum.back());
    }
    auto func = [&](double x)
    {
        return 2 * rankPartialSum[x - 1] - x * (n + 1);
    };
    Eigen::ArrayXd Uk = k.unaryExpr(func);
    int K; // K will take into account 0 numbering
    double U = Uk.abs().maxCoeff(&K);
    double pval = std::min(1.0, 2 * std::exp((-6 * U * U) / (n * n * (n + 1))));
    return std::make_tuple(pval, U, K);
}
std::tuple<double, double, int> Pettitt::test2(std::deque<double> const &x, std::vector<double> const &x2)
{
    size_t n = x.size() + x2.size();
    Eigen::ArrayXd k(n);
    for (size_t idx = 0; idx != n; ++idx)
        k(idx) = idx + 1;

    std::vector<double> rank = ranks2(x, x2);
    std::vector<double> rankPartialSum;
    rankPartialSum.reserve(n);
    rankPartialSum.push_back(rank[0]);
    for (size_t idx = 1; idx != n; ++idx)
    {
        rankPartialSum.push_back(rank[idx] + rankPartialSum.back());
    }
    auto func = [&](double x)
    {
        return 2 * rankPartialSum[x - 1] - x * (n + 1);
    };
    Eigen::ArrayXd Uk = k.unaryExpr(func);
    int K; // K will take into account 0 numbering
    double U = Uk.abs().maxCoeff(&K);
    double pval = std::min(1.0, 2 * std::exp((-6 * U * U) / (n * n * (n + 1))));
    return std::make_tuple(pval, U, K);
}
// pval, U, K in that order
std::tuple<double, double, int> Pettitt::test(std::vector<double> const &x)
{
    size_t n = x.size();
    Eigen::ArrayXd k(n);
    for (size_t idx = 0; idx != n; ++idx)
        k(idx) = idx + 1;

    std::vector<double> rank = ranks(x);
    std::vector<double> rankPartialSum;
    rankPartialSum.reserve(n);
    rankPartialSum.push_back(rank[0]);
    for (size_t idx = 1; idx != n; ++idx)
    {
        rankPartialSum.push_back(rank[idx] + rankPartialSum.back());
    }
    auto func = [&](double x)
    {
        return 2 * rankPartialSum[x - 1] - x * (n + 1);
    };
    Eigen::ArrayXd Uk = k.unaryExpr(func);
    size_t K; // K will take into account 0 numbering
    double U = Uk.abs().maxCoeff(&K);
    double pval = std::min(1.0, 2 * std::exp((-6 * U * U) / (n * n * (n + 1))));
    return std::make_tuple(pval, U, K);
}
std::vector<double> Pettitt::ranks(std::vector<double> const &x)
{
    std::vector<std::tuple<double, size_t, double>> vec;
    vec.reserve(x.size());
    size_t idx = 1;
    for (auto const &item : x)
    {
        vec.emplace_back(item, idx, 0);
        ++idx;
    }
    std::stable_sort(vec.begin(), vec.end(),
                     [](auto const &tup1, auto const &tup2)
                     {
                         return std::get<0>(tup1) < std::get<0>(tup2);
                     });

    size_t const vecSize = vec.size();
    size_t i = 0;
    while (i < vecSize)
    {
        auto &[value, idx1, rank] = vec[i];
        size_t currI = i;
        size_t idxSum = i + 1;
        size_t nrDuplicates = 1;
        ++i;
        while (i < vecSize)
        {
            auto &[value2, idx2, rank2] = vec[i];
            if (value == value2)
            {
                ++nrDuplicates;
                idxSum += i + 1;
            }
            else
            {
                break;
            }
            ++i;
        }
        for (size_t inIdx = currI; inIdx < i; ++inIdx)
        {
            get<2>(vec[inIdx]) = static_cast<double>(idxSum) / nrDuplicates;
        }
    }

    std::stable_sort(vec.begin(), vec.end(),
                     [](auto const &tup1, auto const &tup2)
                     {
                         return std::get<1>(tup1) < std::get<1>(tup2);
                     });
    std::vector<double> rank;
    rank.reserve(vecSize);
    for (auto const &[value, id, rankCurr] : vec)
        rank.push_back(rankCurr);
    return rank;
}
std::vector<double> Pettitt::ranks2(std::vector<double> const &x, std::vector<double> const &x2)
{
    std::vector<std::tuple<double, size_t, double>> vec;
    vec.reserve(x.size() + x2.size());
    size_t idx = 1;
    for (auto const &item : x)
    {
        vec.emplace_back(item, idx, 0);
        ++idx;
    }
    for (auto const &item : x2)
    {
        vec.emplace_back(item, idx, 0);
        ++idx;
    }
    std::stable_sort(vec.begin(), vec.end(),
                     [](auto const &tup1, auto const &tup2)
                     {
                         return std::get<0>(tup1) < std::get<0>(tup2);
                     });

    size_t const vecSize = vec.size();
    size_t i = 0;
    while (i < vecSize)
    {
        auto &[value, idx1, rank] = vec[i];
        size_t currI = i;
        size_t idxSum = i + 1;
        size_t nrDuplicates = 1;
        ++i;
        while (i < vecSize)
        {
            auto &[value2, idx2, rank2] = vec[i];
            if (value == value2)
            {
                ++nrDuplicates;
                idxSum += i + 1;
            }
            else
            {
                break;
            }
            ++i;
        }
        for (size_t inIdx = currI; inIdx < i; ++inIdx)
        {
            get<2>(vec[inIdx]) = static_cast<double>(idxSum) / nrDuplicates;
        }
    }

    std::stable_sort(vec.begin(), vec.end(),
                     [](auto const &tup1, auto const &tup2)
                     {
                         return std::get<1>(tup1) < std::get<1>(tup2);
                     });
    std::vector<double> rank;
    rank.reserve(vecSize);
    for (auto const &[value, id, rankCurr] : vec)
        rank.push_back(rankCurr);
    return rank;
}
std::vector<double> Pettitt::ranks2(std::deque<double> const &x, std::vector<double> const &x2)
{
    std::vector<std::tuple<double, size_t, double>> vec;
    vec.reserve(x.size() + x2.size());
    size_t idx = 1;
    for (auto const &item : x)
    {
        vec.emplace_back(item, idx, 0);
        ++idx;
    }
    for (auto const &item : x2)
    {
        vec.emplace_back(item, idx, 0);
        ++idx;
    }
    std::stable_sort(vec.begin(), vec.end(),
                     [](auto const &tup1, auto const &tup2)
                     {
                         return std::get<0>(tup1) < std::get<0>(tup2);
                     });

    size_t const vecSize = vec.size();
    size_t i = 0;
    while (i < vecSize)
    {
        auto &[value, idx1, rank] = vec[i];
        size_t currI = i;
        size_t idxSum = i + 1;
        size_t nrDuplicates = 1;
        ++i;
        while (i < vecSize)
        {
            auto &[value2, idx2, rank2] = vec[i];
            if (value == value2)
            {
                ++nrDuplicates;
                idxSum += i + 1;
            }
            else
            {
                break;
            }
            ++i;
        }
        for (size_t inIdx = currI; inIdx < i; ++inIdx)
        {
            get<2>(vec[inIdx]) = static_cast<double>(idxSum) / nrDuplicates;
        }
    }

    std::stable_sort(vec.begin(), vec.end(),
                     [](auto const &tup1, auto const &tup2)
                     {
                         return std::get<1>(tup1) < std::get<1>(tup2);
                     });
    std::vector<double> rank;
    rank.reserve(vecSize);
    for (auto const &[value, id, rankCurr] : vec)
        rank.push_back(rankCurr);
    return rank;
}
