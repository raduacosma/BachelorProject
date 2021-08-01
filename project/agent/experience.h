/*
     Copyright (C) 2021  Radu Alexandru Cosma

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef _INCLUDED_EXPERIENCE
#define _INCLUDED_EXPERIENCE

#include "../../Eigen/Core"
struct Experience
{
    size_t action;
    float reward;
    bool isTerminal;
    Eigen::VectorXf lastState;
    Eigen::VectorXf newState;
};
struct OpExperience
{
    Eigen::VectorXf lastState;
    Eigen::VectorXf newState;
};
#endif
