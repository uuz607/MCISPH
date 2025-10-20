#pragma once 
#include <vector>
#include <vector_types.h>
#include "Geometry.cuh"
#include "Common.cuh"
#include "FluidSolver.h"
#include <thrust/host_vector.h>

namespace mcisph
{
	struct FluidGAS
	{
		OptixTraversableHandle  handle;

		OptixBuildInput         build_input;

		uint32_t custom_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

		OptixAabb*				d_aabb_buffer;
		uint32_t                num_primitives;

		OptixAccelBufferSizes   buffer_sizes ;
		CUdeviceptr             d_update_buffer;

		CUdeviceptr             d_output_buffer;
	};

	struct BoundaryGAS
	{
		OptixTraversableHandle  handle;
		CUdeviceptr             d_output_buffer = 0;
	};

	struct SceneIAS
	{
		OptixTraversableHandle  handle;
		OptixBuildInput         build_input;
		CUdeviceptr             d_instances ;
		OptixAccelBufferSizes   buffer_sizes ;
		CUdeviceptr             d_update_buffer;
		CUdeviceptr             d_output_buffer;
	};

	struct FluidOptixState
	{
		OptixDeviceContext context = nullptr;

		Params params;
		CUdeviceptr d_params = 0;

		DeviceFluid     fluids;
		DeviceBoundary  boundary;
		DeviceConfig    config;
		RandState*		d_volume_rand_states;
		RandState*		d_vpl_rand_states;
		VPLRecord*		d_vpl_records;
		uint32_t*		d_Morton_code;

		SceneIAS                       scene_ias = {};
		FluidGAS					   fluids_gas={};
		BoundaryGAS                    boundary_gas = {};

		OptixModule                    fluid_solver_module = nullptr;

		OptixPipelineCompileOptions    pipeline_compile_options = {};

		OptixPipeline 			       neighbor_search_pipeline;
		OptixPipeline				   press_proj_construct_pipeline;

		OptixProgramGroup              raygen_neighbor_search;
		OptixProgramGroup              raygen_press_proj_construct;

		OptixProgramGroup              hit_neighbor_search;
		OptixProgramGroup              hit_sample_vpl_source_point;
		OptixProgramGroup			   hit_sample_vpl_point;
		OptixProgramGroup			   hit_approx_boundary_value;

		OptixProgramGroup              miss_default;

		CUdeviceptr					   d_raygen_neighbor_search_record;
		CUdeviceptr					   d_raygen_proj_construct_record;
	
		CUdeviceptr					   d_hit_neighbor_search_record;
		CUdeviceptr					   d_hit_project_records;

		CUdeviceptr					   d_missgroup_records;

		OptixShaderBindingTable        sbt_press_proj_construct = {};
		OptixShaderBindingTable        sbt_neighbor_search = {};

		int device_count;
		int device_index = 0;
	};

	class FluidModel
	{
	public:
		~FluidModel();
		void setConfig(FluidModelConfig&& host_config, DeviceConfig&& device_config);
		void setFluidParticles(const std::vector<Particle>& particles);
		void setBoundary(Boundary&& solid_boundary);
		void setDevice();

		float getTimeStep();
		const Fluid& getFluid();

		//Optix resources
		void createOptixContext();
		void prepareDataForOptix();

		void genParticleAABB();
		void buildSceneIAS();
		void updateSceneIAS();

		void createMoudle();
		void createProgramGroups();
		void createSBT();
		void createPipeline();
		void initLaunchParams();
		
		//simulation steps
		void neighborSearch();
		void calcDensity();
		void calcNonPressure();

		//adjust the time step according to CFL condition
		void computeTimeStep();
		void predictVel();

		//pressure projection
		void correctVel();
		void calcDivAndNormal();
		void calcVolumeIntegralTerm();
		void calcBoundaryIntegralTerm();

		//update positions
		void updatePos();
		void sortParticlesByMortonCode();

	protected:
		Fluid fluid;
		Boundary solid_boundary;

		FluidOptixState optix_state;

		FluidModelConfig model_params;

		SPHKernelWCoeff wcoeff;
		SPHKernelDWCoeff dwcoeff;

		void cleanupOptixState();

		void buildBoundaryGAS();

		void buildFluidGAS();
		void updateFluidGAS();
	};
}