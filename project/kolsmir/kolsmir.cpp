// Kolmogorov-Smirnov two-sample test, adapted from the ROOT data analysis framework
// (https://root.cern/doc/master/TMath_8cxx_source.html)
// @(#)root/mathcore:$Id$
// Authors: Rene Brun, Anna Kreshuk, Eddy Offermann, Fons Rademakers   29/07/95

/*************************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.                           *
 * All rights reserved.                                                              *
 *                                                                                   *
 * For the licensing terms see LGPLv2_1.                                             *
 * For the list of contributors see CREDITS (copied directly from ROOT repository).  *
 *************************************************************************************/

// Copyright (c) 2021, Radu Alexandru Cosma

/*
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include "kolsmir.h"
#include <cassert>
#include <cmath>
#include <iostream>
double KolSmir::probability(double z)
{
    double fj[4] = { -2, -8, -18, -32 }, r[4];
    double const w = 2.50662827;
    // c1 - -pi**2/8, c2 = 9*c1, c3 = 25*c1
    double const c1 = -1.2337005501361697;
    double const c2 = -11.103304951225528;
    double const c3 = -30.842513753404244;

    double u = std::abs(z);
    double p;
    if (u < 0.2)
    {
        p = 1;
    }
    else if (u < 0.755)
    {
        double v = 1. / (u * u);
        p = 1 - w * (std::exp(c1 * v) + std::exp(c2 * v) + std::exp(c3 * v)) / u;
    }
    else if (u < 6.8116)
    {
        r[1] = 0;
        r[2] = 0;
        r[3] = 0;
        double v = u * u;
        int maxj = std::max(1, nint(3. / u));
        for (int j = 0; j < maxj; j++)
        {
            r[j] = std::exp(fj[j] * v);
        }
        p = 2 * (r[0] - r[1] + r[2] - r[3]);
    }
    else
    {
        p = 0;
    }
    return p;
}

std::tuple<double, double> KolSmir::test(int na, double const *a, int nb, double const *b)
{
    double prob = -1;
    if (!a || !b || na <= 2 || nb <= 2)
    {
        std::cout << "Kolsmir test arrays too small" << '\n';
        return std::make_tuple(prob, -1);
    }
    double rna = na;
    double rnb = nb;
    double sa = 1. / rna;
    double sb = 1. / rnb;
    double rdiff = 0;
    double rdmax = 0;
    int ia = 0;
    int ib = 0;

    //    Main loop over point sets to find max distance
    //    rdiff is the running difference, and rdmax the max.
    bool ok = false;
    for (int i = 0; i < na + nb; i++)
    {
        if (a[ia] < b[ib])
        {
            rdiff -= sa;
            ia++;
            if (ia >= na)
            {
                ok = true;
                break;
            }
        }
        else if (a[ia] > b[ib])
        {
            rdiff += sb;
            ib++;
            if (ib >= nb)
            {
                ok = true;
                break;
            }
        }
        else
        {
            // special cases for the ties
            double x = a[ia];
            while (ia < na && a[ia] == x)
            {
                rdiff -= sa;
                ia++;
            }
            while (ib < nb && b[ib] == x)
            {
                rdiff += sb;
                ib++;
            }
            if (ia >= na)
            {
                ok = true;
                break;
            }
            if (ib >= nb)
            {
                ok = true;
                break;
            }
        }
        rdmax = std::max(rdmax, std::abs(rdiff));
    }
    assert(ok);

    if (ok)
    {
        rdmax = std::max(rdmax, std::abs(rdiff));
        double z = rdmax * std::sqrt(rna * rnb / (rna + rnb));
        prob = probability(z);
    }
    return std::make_tuple(prob, rdmax);
}
