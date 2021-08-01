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

#ifndef _INCLUDED_KOLSMIR
#define _INCLUDED_KOLSMIR
#include <tuple>

class KolSmir
{
  public:
    double probability(double z);
    std::tuple<double, double> test(int na, double const *a, int nb, double const *b);
    template <typename T>
    int nint(T x);
};

template <typename T>
inline int KolSmir::nint(T x)
{
    int i;
    if (x >= 0)
    {
        i = int(x + 0.5);
        if (i & 1 && x + 0.5 == T(i))
            i--;
    }
    else
    {
        i = int(x - 0.5);
        if (i & 1 && x - 0.5 == T(i))
            i++;
    }
    return i;
}
#endif
