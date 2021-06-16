#ifndef _INCLUDED_RUNHEADLESS
#define _INCLUDED_RUNHEADLESS

#include "hyperparamSpec/hyperparamSpec.h"
#include <string>
void runHeadless(std::string const &file);
HyperparamSpec loadHyperparameters(std::string const &file);
#endif
