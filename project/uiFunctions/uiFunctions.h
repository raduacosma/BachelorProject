#ifndef _INCLUDED_UIFUNCTIONS
#define _INCLUDED_UIFUNCTIONS

#include "../simState/simState.h"
#include "../simBuilder/simBuilder.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#undef IMGUI_DEFINE_PLACEMENT_NEW
#define IMGUI_DEFINE_PLACEMENT_NEW
#undef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS

#include "imgui_internal.h"


#include <iostream>

template <typename Repr>
concept StateRepresentation = requires(Repr repr)
{
    repr.getWidth();
    repr.getHeight();
    repr.getFullMazeRepr();
};


void drawMenuBar(SimBuilder &simBuilder);
void drawMenuBar(SimState const &simState);

template <typename StateRepr>
void drawGameState(StateRepr const &simState)
{
    ImGuiViewportP *viewport =
        (ImGuiViewportP *)(void *)ImGui::GetMainViewport();
    ImVec2 menu_bar_pos = viewport->Pos + viewport->CurrWorkOffsetMin;
    ImVec2 menu_bar_size =
        ImVec2(viewport->Size.x - viewport->CurrWorkOffsetMin.x +
               viewport->CurrWorkOffsetMax.x,
               1.0f);
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin(
        "MainWindow", 0,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse);
    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos(); // ImDrawList API uses
    // screen coordinates!
    ImVec2 canvas_sz =
        ImGui::GetContentRegionAvail(); // Resize canvas to what's available
    if (canvas_sz.x < 50.0f)
        canvas_sz.x = 50.0f;
    if (canvas_sz.y < 50.0f)
        canvas_sz.y = 50.0f;
    ImVec2 canvas_p1 =
        ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
    draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

    renderSimState(simState, canvas_p0, canvas_p1, draw_list);
    ImGui::End();
}





template <StateRepresentation StateRepr>
void renderSimState(StateRepr const &simState, ImVec2 const &canvas_p0,
                    ImVec2 const &canvas_p1, ImDrawList *draw_list)
{
    float xStepSize = (canvas_p1.x - canvas_p0.x) / simState.getWidth();
    float yStepSize = (canvas_p1.y - canvas_p0.y) / simState.getHeight();
    float xCurr = canvas_p0.x;
    float xInitial = xCurr;
    float yCurr = canvas_p0.y;
    auto stateRepr = simState.getFullMazeRepr();
    for (size_t i = 0; i < simState.getHeight(); ++i)
    {
        for (size_t j = 0; j < simState.getWidth(); ++j)
        {
            TileStates currTile = stateRepr[i][j];
            auto drawRect = [&](size_t r, size_t g, size_t b, size_t a){
              draw_list->AddRectFilled({ xCurr+2, yCurr+2 },
                                       { xCurr + xStepSize - 2, yCurr + yStepSize - 2},
                                       IM_COL32(r,g,b,a));
            };
            switch (currTile)
            {
                case TileStates::EMPTY:
                    drawRect(30,30,30,255);
                    break;
                case TileStates::AGENT:
                    drawRect(60,60,60,255);
                    break;
                case TileStates::AGENT_TRACE:
                    drawRect(80,80,80,255);
                    break;
                case TileStates::OPPONENT:
                    drawRect(100,100,100,255);
                    break;
                case TileStates::OPPONENT_TRACE:
                    drawRect(120,120,120,255);
                    break;
                case TileStates::WALL:
                    drawRect(140,140,140,255);
                    break;
                case TileStates::GOAL:
                    drawRect(160,160,160,255);
                    break;
            }

            xCurr += xStepSize;
        }
        xCurr = xInitial;
        yCurr += yStepSize;
    }
}


#endif
