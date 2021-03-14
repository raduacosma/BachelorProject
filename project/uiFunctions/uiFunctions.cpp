#include "uiFunctions.h"

using namespace std;
void drawMenuBar(SimBuilder &simBuilder)
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

        ImGui::EndMainMenuBar();
    }
}
void drawMenuBar(SimState const &simState)
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::Button("Pause simulation"))
        {
            std::cout << "yues" << std::endl;
        }

        ImGui::EndMainMenuBar();
    }
}
void drawSizeSelectionMenu(UiStateTracker &uiStateTracker)
{
    ImGuiContext& g = *GImGui;
    ImGui::SetNextWindowPos(g.IO.DisplaySize * 0.5f, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize({0,0});
    ImGui::Begin("Size Selector");

    ImGui::Text("Please select the cell width and height of the next simulation");
    static string width{"17"};
    ImGui::InputText("World Width", &width);
    static string height{"10"};
    ImGui::InputText("World Height", &height);
    if(ImGui::Button("Apply"))
    {
        uiStateTracker.nextSimCellWidth = stoul(width);
        uiStateTracker.nextSimCellHeight = stoul(height);
        uiStateTracker.showSizeSelectionMenu = false;
    }
    ImGui::End();

}
void updateSimBuilder(SimBuilder &simBuilder)
{
    ImVec2 mousePos = ImGui::GetMousePos();
    if(mousePos.x < simBuilder.canvasBegPos.x or mousePos.x > simBuilder.canvasEndPos.x or
        mousePos.y < simBuilder.canvasBegPos.y or mousePos.y > simBuilder.canvasEndPos.y)
        return;
    if(ImGui::IsMouseDragging(0) or ImGui::IsMouseClicked(0))
    {
        simBuilder.drawAtPos(
            {static_cast<size_t>((mousePos - simBuilder.canvasBegPos).x / simBuilder.canvasStepSize.x),
            static_cast<size_t>((mousePos - simBuilder.canvasBegPos).y / simBuilder.canvasStepSize.y)});
    }
    if (ImGui::IsMouseDragging(1) or ImGui::IsMouseClicked(1))
    {
        simBuilder.removeAtPos(
            {static_cast<size_t>((mousePos - simBuilder.canvasBegPos).x / simBuilder.canvasStepSize.x),
             static_cast<size_t>((mousePos - simBuilder.canvasBegPos).y / simBuilder.canvasStepSize.y)});
    }
}
