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