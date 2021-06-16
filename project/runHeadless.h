#ifndef _INCLUDED_RUNHEADLESS
#define _INCLUDED_RUNHEADLESS

#include <string>
#include "hyperparamSpec/hyperparamSpec.h"
void runHeadless(std::string const &file);
HyperparamSpec loadHyperparameters(std::string const &file);
#endif
