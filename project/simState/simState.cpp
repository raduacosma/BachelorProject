#include "simState.h"
using namespace std;

std::vector <std::vector<TileStates>> SimState::getFullMazeRepr() const
{
    for(size_t i = 0; i!=simSize.y; ++i)
        for(size_t j = 0; j!=simSize.x; ++j)
        {

        }
    vector<TileStates> row{simSize.x, TileStates::EMPTY};
    return vector<vector<TileStates>>{simSize.y,row};

}

SimState::SimState(size_t width, size_t height)
:
    simSize({width,height})
{
}
