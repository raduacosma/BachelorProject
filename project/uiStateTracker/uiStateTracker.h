#ifndef _INCLUDED_UISTATETRACKER
#define _INCLUDED_UISTATETRACKER


struct UiStateTracker
{
    size_t nextSimCellWidth = 10;
    size_t nextSimCellHeight = 17;
    bool showSizeSelectionMenu = true;
    bool showSimBuilder = false;
    bool showSimState = false;
};
#endif
