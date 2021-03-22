#ifndef _INCLUDED_UIFUNCTIONS
#define _INCLUDED_UIFUNCTIONS

#include "../simState/simState.h"
#include "../simBuilder/simBuilder.h"
#include "../uiStateTracker/uiStateTracker.h"


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "../imgui-1.81/misc/cpp/imgui_stdlib.h"

#undef IMGUI_DEFINE_PLACEMENT_NEW
#define IMGUI_DEFINE_PLACEMENT_NEW
#undef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS

#include "../simContainer/simContainer.h"
#include "imgui_internal.h"

#include <iostream>

template <typename Repr>
concept StateRepresentation = requires(Repr repr)
{
    repr.getWidth();
    repr.getHeight();
    repr.getFullMazeRepr();
};

void drawStartMenu(UiStateTracker &uiStateTracker);
void drawMenuBar(SimBuilder &simBuilder, UiStateTracker &uiStateTracker);
void drawMenuBar(SimContainer const &simState, UiStateTracker &uiStateTracker);

void updateSimBuilder(SimBuilder &simBuilder);

template <typename StateRepr>
void drawGameState(StateRepr &simState)
{
    ImGuiViewportP *viewport =
        (ImGuiViewportP *)(void *)ImGui::GetMainViewport();

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
void renderSimState(StateRepr &simState, ImVec2 const &canvas_p0,
                    ImVec2 const &canvas_p1, ImDrawList *draw_list)
{
    float xStepSize = (canvas_p1.x - canvas_p0.x) / simState.getWidth();
    float yStepSize = (canvas_p1.y - canvas_p0.y) / simState.getHeight();
    simState.updateCanvasStepSize({xStepSize,yStepSize});
    simState.updateCanvasBegPos({ canvas_p0.x, canvas_p0.y });
    simState.updateCanvasEndPos({canvas_p1.x,canvas_p1.y});

    float xCurr = canvas_p0.x;
    float xInitial = xCurr;
    float yCurr = canvas_p0.y;
    float yInitial = yCurr;
    auto stateRepr = simState.getFullMazeRepr();
    for (size_t i = 0; i < simState.getWidth(); ++i)
    {
        for (size_t j = 0; j < simState.getHeight(); ++j)
        {
            ImVec4 currTile = stateRepr[i][j];
            auto drawRect = [&](ImVec4 color){
              draw_list->AddRectFilled({ xCurr+2, yCurr+2 },
                                       { xCurr + xStepSize - 2, yCurr + yStepSize - 2},
                                       IM_COL32(color.x,color.y,color.z,color.w));
            };
            drawRect(currTile);


            yCurr += yStepSize;
        }
        yCurr = yInitial;
        xCurr += xStepSize;
    }
}


#endif
