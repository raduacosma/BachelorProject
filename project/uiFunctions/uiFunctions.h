/*
The MIT License (MIT)

Copyright (c) 2014-2021 Omar Cornut
Copyright (c) 2021 Radu Alexandru Cosma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

// Some code for drawing the GUI was adapted from the examples of ImGUI (https://github.com/ocornut/imgui)

#if SHOULD_HAVE_GUI
#ifndef _INCLUDED_UIFUNCTIONS
#define _INCLUDED_UIFUNCTIONS

#include "../simBuilder/simBuilder.h"
#include "../simState/simState.h"
#include "../uiStateTracker/uiStateTracker.h"

#include "../imgui-1.81/misc/cpp/imgui_stdlib.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

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
    ImGuiViewportP *viewport = (ImGuiViewportP *)(void *)ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("MainWindow", 0,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoScrollWithMouse);
    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
    ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
    if (canvas_sz.x < 50.0f)
        canvas_sz.x = 50.0f;
    if (canvas_sz.y < 50.0f)
        canvas_sz.y = 50.0f;
    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
    draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

    renderSimState(simState, canvas_p0, canvas_p1, draw_list);
    ImGui::End();
}

template <StateRepresentation StateRepr>
void renderSimState(StateRepr &simState, ImVec2 const &canvas_p0, ImVec2 const &canvas_p1, ImDrawList *draw_list)
{
    float xStepSize = (canvas_p1.x - canvas_p0.x) / simState.getWidth();
    float yStepSize = (canvas_p1.y - canvas_p0.y) / simState.getHeight();
    simState.updateCanvasStepSize({ xStepSize, yStepSize });
    simState.updateCanvasBegPos({ canvas_p0.x, canvas_p0.y });
    simState.updateCanvasEndPos({ canvas_p1.x, canvas_p1.y });

    float xCurr = canvas_p0.x;
    float yCurr = canvas_p0.y;
    float yInitial = yCurr;
    auto stateRepr = simState.getFullMazeRepr();
    for (size_t i = 0; i < simState.getWidth(); ++i)
    {
        for (size_t j = 0; j < simState.getHeight(); ++j)
        {
            FloatVec4 currTile = stateRepr[i][j];
            auto drawRect = [&](FloatVec4 color)
            {
                draw_list->AddRectFilled({ xCurr + 2, yCurr + 2 }, { xCurr + xStepSize - 2, yCurr + yStepSize - 2 },
                                         IM_COL32(color.x, color.y, color.z, color.w));
            };
            drawRect(currTile);

            yCurr += yStepSize;
        }
        yCurr = yInitial;
        xCurr += xStepSize;
    }
}

#endif
#endif
