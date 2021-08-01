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

#if SHOULD_HAVE_GUI
#ifndef _INCLUDED_UISTATETRACKER
#define _INCLUDED_UISTATETRACKER

struct UiStateTracker
{
    std::string nextFilename;
    size_t nextSimCellWidth = 10;
    size_t nextSimCellHeight = 17;
    bool showStartMenu = true;
    bool showSimBuilder = false;
    bool showSimState = false;
    bool gamePaused = true;
    bool playOneStep = false;
};
#endif
#endif