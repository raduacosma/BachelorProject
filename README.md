This is the source code used for my Bachelor's Project "Moving object recognition and avoidance using
reinforcement learning".

It includes everything required to reproduce the results:
- implementations for Multi-Layer Perceptrons, Sarsa, Q-learning with Experience Replay and Double Q-learning with Experience Replay.

- implementations for the novel opponent recognition techniques presented in the paper, and two statistical procedures: Pettitt's test for Change-Point detection and the Kolmogorov-Smirnov two-sample test.

- a graphical simulation of the game and the simulation code for the level-based game itself.

- hyperparameter optimisation procedure, random seeds used, data preprocessing and plotting code used to obtain the results

The random seeds are fixed and therefore the program is deterministic. The C++ Mersenne Twister random number generator was used which should provide the same results with all compilers. However, the C++ standard library distributions were used which are not the same over all compilers and platforms and thus the exact numbers in the thesis might not be fully reproducible (though of course, very similar results will be obtained when attempting to reproduce).

In order to compile the code, a C++20-compliant compiler is required. The results in the thesis were produced with g++ 10.2.0.

You can compile the code using the included CMake files. For the graphical simulation, the GLFW library needs to be installed as the code dinamically links to it. If the graphical simulation is not desired (and thus GLFW does not need to be installed), you can simply set the SHOULD_HAVE_GUI variable to FALSE. This FALSE value was also used when producing the results in the thesis.

Everything is self-contained in this project, there are no external dependencies to reproduce the results (unless GLFW, but that is only if the graphical simulation is desired).

The following code was used from other sources:
- The Eigen matrix linear algebra library (included in the code so that no dependencies are required)
- The ImGUI GUI library (included in the code so that no dependencies are required)

The following code was adapted from other sources:
- ImGUI tutorial samples were adapted for drawing a GUI
- The Kolmogorov-Smirnov two-sample test was adapted from the ROOT data analysis framework
- Pettitt's test for change-point detection was rewritten from the "trend" package from R into C++

All the code that is or modifies the source code from the other sources mentioned above has the same license as that code. All other code is GPLv3 for compatibility with the "trend" package code which is also GPLv3. According to the licenses of the project the whole project is licensed under GPLv3 with the abovementioned libraries and other source code licensed with their original licenses. This statically linked approach was taken so that the results are easily reproducible (no library change affects the code and only a compiler is needed to run the code).


How to run:

```shell
mkdir build
cd build
cmake ..
make
cd ../exeFolder # if running the main code, 
                # do not do this for the 
                # data preprocessing and 
                # hyperparameter generation scripts
./name_of_executable # for example, BachelorProject
                     # also add command line parameters
                     # where needed
```

For the main project (the BachelorProject executable),
the first command line parameter necessary is the file name
of the configuration to be used. Such a file can be found in exeFolder/initialParameterValuesComplexGame.txt. The valid ranges for the hyperparameters can be found in the thesis and also in hyperparameterGeneration/generateHyperparams.cpp. Of note is that the enumerations used (learning algorithm, opponent modelling technique) are encoded in the config files with the 0-starting position in the array for that component specified in hyperparameterGeneration/generateHyperparams.cpp.

The second command line parameter is optional and if present, will start a GUI simulation if the project was compiled with SHOULD_HAVE_GUI as TRUE (and thus GLFW is also present, as otherwise it would not have compiled).

There are a number of ways to print results, including final statistics (presented in the thesis) and statistics per episode (presented as plots in the thesis). You can specify the printing function in project/runHeadless.cpp. Currently, the statistics per episode function is used.

I would like to thank the Center for Information Technology of the University of Groningen for their support and for providing access to the Peregrine high performance computing cluster.

