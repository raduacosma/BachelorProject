#include "uiFunctions.h"


void drawMenuBar(SimBuilder &simBuilder)
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::Button("Draw Agent"))
        {
        }
        if (ImGui::Button("Draw Opponent"))
        {
        }
        if (ImGui::Button("Draw Opponent Path"))
        {
        }
        if (ImGui::Button("Draw Wall"))
        {
        }
        if (ImGui::Button("Draw Goal"))
        {
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


