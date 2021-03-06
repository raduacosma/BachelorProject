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
#include "uiFunctions.h"

using namespace std;
void drawMenuBar(SimBuilder &simBuilder, UiStateTracker &uiStateTracker)
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::Button("Draw Agent"))
        {
            simBuilder.objToDraw = SimObject::AGENT;
        }
        if (ImGui::Button("Draw Opponent"))
        {
            simBuilder.objToDraw = SimObject::OPPONENT;
        }
        if (ImGui::Button("Draw Opponent Path"))
        {
            simBuilder.objToDraw = SimObject::OPPONENT_TRACE;
        }
        if (ImGui::Button("Draw Wall"))
        {
            simBuilder.objToDraw = SimObject::WALL;
        }
        if (ImGui::Button("Draw Goal"))
        {
            simBuilder.objToDraw = SimObject::GOAL;
        }
        if (ImGui::Button("Save model"))
        {

            ImGui::OpenPopup("Save Dialog");
        }
        if (ImGui::Button("Go back to start"))
        {
            uiStateTracker.showSimBuilder = false;
            uiStateTracker.showStartMenu = true;
        }
        ImGuiContext &g = *GImGui;
        ImGui::SetNextWindowPos(g.IO.DisplaySize * 0.5f, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize({ 0, 0 });
        if (ImGui::BeginPopupModal("Save Dialog", NULL))
        {
            simBuilder.objToDraw = SimObject::NONE;
            ImGui::Text("Please select the file name of the simulation state");
            static string simName{ "simulation_state.txt" };
            ImGui::InputText("Simulation state file name", &simName);
            if (ImGui::Button("Save"))
            {
                simBuilder.writeToFile(simName);
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel"))
            {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        ImGui::EndMainMenuBar();
    }
}
void drawMenuBar(SimContainer const &simContainer, UiStateTracker &uiStateTracker)
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::Button("Play / Pause"))
        {
            uiStateTracker.gamePaused = !uiStateTracker.gamePaused;
        }
        if (ImGui::Button("Step through"))
        {
            if (uiStateTracker.gamePaused)
                uiStateTracker.playOneStep = true;
        }
        string levelText = "Level: " + to_string(simContainer.getCurrSimState());
        ImGui::Text("%s", levelText.c_str());
        string episodeText = "Episode: " + to_string(simContainer.getEpisodeCount());
        ImGui::Text("%s", episodeText.c_str());
        string rewardText = "Last Reward: " + to_string(simContainer.getLastReward());
        ImGui::Text("%s", rewardText.c_str());
        ImGui::EndMainMenuBar();
    }
}
void drawStartMenu(UiStateTracker &uiStateTracker)
{
    ImGuiContext &g = *GImGui;
    ImGui::SetNextWindowPos(g.IO.DisplaySize * 0.5f, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize({ 0, 0 });
    ImGui::Begin("Start Menu");
    static string fileName;
    ImGui::Text("Load simulation state from file:");
    ImGui::InputText("File name", &fileName);
    if (ImGui::Button("Load"))
    {
        uiStateTracker.nextFilename = fileName;
        uiStateTracker.showStartMenu = false;
        uiStateTracker.showSimState = true;
    }

    ImGui::Text("Or prototype on a simulation state; please select the cell width and height of the next simulation");
    static string width{ "25" };
    ImGui::InputText("World Width", &width);
    static string height{ "11" };
    ImGui::InputText("World Height", &height);
    if (ImGui::Button("Apply"))
    {
        uiStateTracker.nextSimCellWidth = stoul(width);
        uiStateTracker.nextSimCellHeight = stoul(height);
        uiStateTracker.showStartMenu = false;
        uiStateTracker.showSimBuilder = true;
    }
    ImGui::End();
}
void updateSimBuilder(SimBuilder &simBuilder)
{
    ImVec2 mousePos = ImGui::GetMousePos();
    if (mousePos.x < simBuilder.canvasBegPos.x or mousePos.x > simBuilder.canvasEndPos.x or
        mousePos.y < simBuilder.canvasBegPos.y or mousePos.y > simBuilder.canvasEndPos.y)
        return;
    if (ImGui::IsMouseDragging(0) or ImGui::IsMouseClicked(0))
    {
        simBuilder.drawAtPos(
            { static_cast<size_t>((mousePos.x - simBuilder.canvasBegPos.x) / simBuilder.canvasStepSize.x),
              static_cast<size_t>((mousePos.y - simBuilder.canvasBegPos.y) / simBuilder.canvasStepSize.y) });
    }
    if (ImGui::IsMouseDragging(1) or ImGui::IsMouseClicked(1))
    {
        simBuilder.removeAtPos(
            { static_cast<size_t>((mousePos.x - simBuilder.canvasBegPos.x) / simBuilder.canvasStepSize.x),
              static_cast<size_t>((mousePos.y - simBuilder.canvasBegPos.y) / simBuilder.canvasStepSize.y) });
    }
}
#endif
