#include "MCISPH.h"

int main(int argc, char* argv[])
{	
	const char* json = argv[1];
	std::string config = getConfigPath() + json;

	mcisph::MCISPH simulator;
	simulator.loadConfig(config);
	simulator.initFluidModel();
	simulator.simulate();
}