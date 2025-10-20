#pragma once
#include "Common.cuh"
#include "FluidModel.h"

const std::string getConfigPath();

namespace mcisph
{
	class MCISPH
	{
	public:
		void loadConfig(std::string json);
		void initFluidModel();
		void simulate();
	private:
		void loadFluid();
		void loadBoundary();
		void saveFluidState();

		FluidModel fluid_model;

		std::string output_dir;
		std::string bgeo_file;
		std::string obj_file; 

		float current_t = 0.f;
        float max_time;
		float frame_dt;
		uint32_t time_steps = 0;
	};
}