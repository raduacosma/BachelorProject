// Dear ImGui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context
// creation, etc.) If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs


#include "agent/agent.h"
#include "agent/dqerQueueLearning/dqerQueueLearning.h"
#include "agent/qerqueueLearning/qerQueueLearning.h"
#include "agent/sarsa/sarsa.h"
#include "runHeadless.h"
#include "simContainer/simContainer.h"
#include <iostream>
#include <memory>
#include <string>

#if SHOULD_HAVE_GUI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>


#include "uiFunctions/uiFunctions.h"
#include "uiStateTracker/uiStateTracker.h"


#include <fenv.h>
// About Desktop OpenGL function loaders:
//  Modern desktop OpenGL doesn't have a standard portable header file to load OpenGL function pointers.
//  Helper libraries are often used for this purpose! Here we are supporting a few common ones (gl3w, glew, glad).
//  You may use another loader/header of your choice (glext, glLoadGen, etc.), or chose to manually implement your own.
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h> // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h> // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h> // Initialize with gladLoadGL()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD2)
#include <glad/gl.h> // Initialize with gladLoadGL(...) or gladLoaderLoadGL()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
#define GLFW_INCLUDE_NONE      // GLFW including OpenGL headers causes ambiguity or multiple definition errors.
#include <glbinding/Binding.h> // Initialize with glbinding::Binding::initialize()
#include <glbinding/gl/gl.h>
using namespace gl;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
#define GLFW_INCLUDE_NONE // GLFW including OpenGL headers causes ambiguity or multiple definition errors.
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h> // Initialize with glbinding::initialize()
using namespace gl;
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and
// compatibility with old VS compilers. To link with VS2010-era libraries, VS2015+ requires linking with
// legacy_stdio_definitions.lib, which we do using this pragma. Your own project should not be affected, as you are
// likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}
#endif
int main(int argc, char **argv)
{
    //    Eigen::setNbThreads(1);
//    std::cout << "nthreads: " << Eigen::nbThreads() << '\n';
    //    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    if (argc == 2)
    {
        runHeadless(std::string{ argv[1] });
        return 0;
    }
#if SHOULD_HAVE_GUI
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

        // Decide GL+GLSL versions
#ifdef __APPLE__
    // GL 3.2 + GLSL 150
    const char *glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    size_t const windowWidth = 1280;
    size_t const windowHeight = 720;
    GLFWwindow *window = glfwCreateWindow(windowWidth, windowHeight, "Simulation", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    bool err = gladLoadGL() == 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD2)
    bool err = gladLoadGL(glfwGetProcAddress) ==
               0; // glad2 recommend using the windowing library loader instead of the (optionally) bundled one.
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
    bool err = false;
    glbinding::Binding::initialize();
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
    bool err = false;
    glbinding::initialize(
        [](const char *name)
        {
            return (glbinding::ProcAddress)glfwGetProcAddress(name);
        });
#else
    bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of
                      // initialization.
#endif
    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    ImGui::StyleColorsDark();
    //    ImGui::StyleColorsClassic();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    UiStateTracker uiStateTracker;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    std::unique_ptr<SimBuilder> simBuilder = nullptr;
    std::unique_ptr<SimContainer> simContainer = nullptr;
    size_t simSpeed = 10;
    size_t speedCounter = 0;

    HyperparamSpec hs = loadHyperparameters(std::string{ argv[1] });
    size_t nrEpisodesToEpsilonZero = hs.numberOfEpisodes / 4 * 3;

    size_t agentVisionGridArea = hs.agentVisionGridSize * 2 + 1;
    agentVisionGridArea *= agentVisionGridArea;

    size_t opponentVisionGridArea = hs.opponentVisionGridSize * 2 + 1;
    opponentVisionGridArea *= opponentVisionGridArea;
    globalRng = RandObj(hs.seed, -1, 1, hs.sizeExperience);
    OpModellingType opModellingType = hs.opModellingType;
    ExpReplayParams expReplayParams{ .cSwapPeriod = hs.swapPeriod,
                                     .miniBatchSize = hs.miniBatchSize,
                                     .sizeExperience = hs.sizeExperience };
    AgentMonteCarloParams agentMonteCarloParams{ .maxNrSteps = hs.maxNrSteps, .nrRollouts = hs.nrRollouts };
    MLPParams agentMLP{ .sizes = { agentVisionGridArea * 2 + 4, 200, 4 },
                        .learningRate = hs.agentLearningRate,
                        .regParam = hs.agentRegParam,
                        .outputActivationFunc = ActivationFunction::LINEAR,
                        .miniBatchSize = hs.miniBatchSize,
                        .randInit = false };
    MLPParams opponentMLP{ .sizes = { opponentVisionGridArea * 3, 200, 4 },
                           .learningRate = hs.opponentLearningRate,
                           .regParam = hs.opponentRegParam,
                           .outputActivationFunc = ActivationFunction::SOFTMAX,
                           .miniBatchSize = hs.miniBatchSize,
                           .randInit = false };
    Rewards rewards = {
        .normalReward = -0.01f, .killedByOpponentReward = -1.0f, .outOfBoundsReward = -0.01f, .reachedGoalReward = 1.0f
    };
    SimStateParams simStateParams = { .traceSize = hs.traceSize,
                                      .agentVisionGridSize = hs.agentVisionGridSize,
                                      .opponentVisionGridSize = hs.opponentVisionGridSize,
                                      .randomOpCoef = 0.2 };
    OpTrackParams opTrackParams = { .pValueThreshold = hs.pValueThreshold,
                                    .minHistorySize = hs.minHistorySize,
                                    .maxHistorySize = hs.maxHistorySize };
    std::unique_ptr<Agent> agent;
    switch (hs.agentType)
    {

        case AgentType::SARSA:
            agent = std::make_unique<Sarsa>(opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP,
                                            hs.numberOfEpisodes, nrEpisodesToEpsilonZero, opModellingType, hs.epsilon,
                                            hs.gamma);
            break;
        case AgentType::DEEPQLEARNING:
            agent = std::make_unique<QERQueueLearning>(opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP,
                                                       expReplayParams, hs.numberOfEpisodes, nrEpisodesToEpsilonZero,
                                                       opModellingType, hs.epsilon, hs.gamma);
            break;
        case AgentType::DOUBLEDEEPQLEARNING:
            agent = std::make_unique<DQERQueueLearning>(opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP,
                                                        expReplayParams, hs.numberOfEpisodes, nrEpisodesToEpsilonZero,
                                                        opModellingType, hs.epsilon, hs.gamma);
            break;
    }

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (uiStateTracker.showStartMenu)
        {
            if (simBuilder)
            {
                simBuilder = nullptr;
            }
            if (simContainer)
            {
                simContainer = nullptr;
                switch (hs.agentType)
                {

                    case AgentType::SARSA:
                        agent = std::make_unique<Sarsa>(opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP,
                                                        hs.numberOfEpisodes, nrEpisodesToEpsilonZero, opModellingType,
                                                        hs.epsilon, hs.gamma);
                        break;
                    case AgentType::DEEPQLEARNING:
                        agent = std::make_unique<QERQueueLearning>(
                            opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP, expReplayParams,
                            hs.numberOfEpisodes, nrEpisodesToEpsilonZero, opModellingType, hs.epsilon, hs.gamma);
                        break;
                    case AgentType::DOUBLEDEEPQLEARNING:
                        agent = std::make_unique<DQERQueueLearning>(
                            opTrackParams, agentMonteCarloParams, agentMLP, opponentMLP, expReplayParams,
                            hs.numberOfEpisodes, nrEpisodesToEpsilonZero, opModellingType, hs.epsilon, hs.gamma);
                        break;
                }
            }
            drawStartMenu(uiStateTracker);
        }
        if (uiStateTracker.showSimBuilder)
        {
            if (simBuilder == nullptr)
            {
                simBuilder =
                    std::make_unique<SimBuilder>(uiStateTracker.nextSimCellWidth, uiStateTracker.nextSimCellHeight);
            }
            drawMenuBar(*simBuilder, uiStateTracker);
            drawGameState(*simBuilder);
            updateSimBuilder(*simBuilder);
        }
        if (uiStateTracker.showSimState)
        {
            speedCounter += simSpeed;

            if (not simContainer)
            {
                simContainer =
                    std::make_unique<SimContainer>(uiStateTracker.nextFilename, agent.get(), rewards, simStateParams);
                agent->newEpisode();
            }
            if (speedCounter >= 60)
            {
                if (not uiStateTracker.gamePaused)
                    agent->performOneStep();
                speedCounter = 0;
            }
            if (uiStateTracker.playOneStep)
            {
                agent->performOneStep();
                uiStateTracker.playOneStep = false;
            }
            drawMenuBar(*simContainer, uiStateTracker);
            drawGameState(simContainer->getCurrentLevel());
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
#endif
    return 0;
}
