#include "FluidModel.h"
#include "Utils/vec_math.h"
#include "Utils/MortonCode.h"
#include "Common.cuh"

#include <iomanip>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/scan.h>

#include "Partio.h"

namespace mcisph
{
    template <typename T>
    struct SbtRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    struct EmptyData {};
    using EmptyRecord = SbtRecord<EmptyData>;

    extern "C" char FluidSolver_ptx[];

    inline static size_t round(size_t x, size_t y) { return ((x + y - 1) / y) * y; }

    void initRandomStates(RandState* d_random_states, uint32_t count)
    {
        std::mt19937_64 random;

        thrust::host_vector<uint64_t> h_random_seeds(count);
        for (auto& seed : h_random_seeds)
            seed = random();

        thrust::device_vector<uint64_t> d_random_seeds = h_random_seeds;

        uint64_t* random_seeds_ptr = d_random_seeds.data().get();

        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(count),
            [=] __device__(uint32_t idx) {
            random_init(random_seeds_ptr[idx], d_random_states[idx]);
        });
    }

    FluidModel::~FluidModel()
    {
        cleanupOptixState();
    }

    void FluidModel::cleanupOptixState()
    {   
        //destory pipeline
        OPTIX_CHECK(optixPipelineDestroy(optix_state.neighbor_search_pipeline));
        OPTIX_CHECK(optixPipelineDestroy(optix_state.press_proj_construct_pipeline));

        CUDA_CHECK(cudaFree((void*)optix_state.d_params));

        //destory programs groups
        OPTIX_CHECK(optixProgramGroupDestroy(optix_state.raygen_neighbor_search));
        OPTIX_CHECK(optixProgramGroupDestroy(optix_state.raygen_press_proj_construct));
        OPTIX_CHECK(optixProgramGroupDestroy(optix_state.hit_neighbor_search));
        OPTIX_CHECK(optixProgramGroupDestroy(optix_state.hit_sample_vpl_source_point));
        OPTIX_CHECK(optixProgramGroupDestroy(optix_state.hit_sample_vpl_point));
        OPTIX_CHECK(optixProgramGroupDestroy(optix_state.hit_approx_boundary_value));
        OPTIX_CHECK(optixProgramGroupDestroy(optix_state.miss_default));

        OPTIX_CHECK(optixModuleDestroy(optix_state.fluid_solver_module));

        OPTIX_CHECK(optixDeviceContextDestroy(optix_state.context));

        //free device resources associated with sbt records
        CUDA_CHECK(cudaFree((void*)optix_state.d_raygen_neighbor_search_record));
        CUDA_CHECK(cudaFree((void*)optix_state.d_raygen_proj_construct_record));
        CUDA_CHECK(cudaFree((void*)optix_state.d_hit_neighbor_search_record));
        CUDA_CHECK(cudaFree((void*)optix_state.d_hit_project_records));
        CUDA_CHECK(cudaFree((void*)optix_state.d_missgroup_records));

        //free device resources associated with fluid particles
        CUDA_CHECK(cudaFreeHost((void*)fluid.h_particles));
        CUDA_CHECK(cudaFree((void*)optix_state.fluids.d_particles));
        CUDA_CHECK(cudaFree((void*)optix_state.fluids.d_neighbor_id));

        //free device resources associated with boundaries
        CUDA_CHECK(cudaFree((void*)optix_state.boundary.d_vertex_buf));
        CUDA_CHECK(cudaFree((void*)optix_state.boundary.d_index_buf));
        CUDA_CHECK(cudaFree((void*)optix_state.boundary.d_normal_buf));
        
        CUDA_CHECK(cudaFree((void*)optix_state.d_volume_rand_states));
        
        //free device resources associated with vpl
        CUDA_CHECK(cudaFree((void*)optix_state.d_vpl_rand_states));
        CUDA_CHECK(cudaFree((void*)optix_state.d_vpl_records));

        //free device resources associated with acceleration structures
        CUDA_CHECK(cudaFree((void*)optix_state.scene_ias.d_output_buffer));
        CUDA_CHECK(cudaFree((void*)optix_state.fluids_gas.d_update_buffer));
        CUDA_CHECK(cudaFree((void*)optix_state.fluids_gas.d_aabb_buffer));
        CUDA_CHECK(cudaFree((void*)optix_state.fluids_gas.d_output_buffer));
        CUDA_CHECK(cudaFree((void*)optix_state.boundary_gas.d_output_buffer));

        CUDA_CHECK(cudaFree((void*)optix_state.d_Morton_code));
    }

    void FluidModel::setConfig(FluidModelConfig&& host_config, DeviceConfig&& device_config)
    {
        model_params = std::move(host_config);
        optix_state.config = std::move(device_config);

        fluid.dt = 0.f;

        fluid.density0 = model_params.density0;
        float d = 2.0f * model_params.particle_radius;
        float v = d * d * d;
        fluid.mass0 = v * fluid.density0;
        
        float h = 4.f * model_params.particle_radius;
        getWCoeff(h, wcoeff);
        getDWCoeff(h, dwcoeff);
    }

    void FluidModel::setBoundary(Boundary&& solid_boundary)
    {
        this->solid_boundary = std::move(solid_boundary);
    }

    void FluidModel::setFluidParticles(const std::vector<Particle>& particle)
    {
        fluid.size = particle.size();
        fluid.count = fluid.size * sizeof(Particle);

        CUDA_CHECK(cudaMallocHost((void**)&fluid.h_particles, fluid.count));
        CUDA_CHECK(cudaMemcpy((void*)fluid.h_particles, (void*)particle.data(), fluid.count, cudaMemcpyHostToHost));
        
        optix_state.config.num_path_sample = fluid.size;
    }

    void FluidModel::setDevice()
    {
        CUDA_CHECK(cudaGetDeviceCount(&optix_state.device_count));
        printf(" Total GPUs visible:%d\n", optix_state.device_count);

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, optix_state.device_index));
        CUDA_CHECK(cudaSetDevice(optix_state.device_index));
    }

    const Fluid& FluidModel::getFluid()
    {
        CUDA_CHECK(cudaMemcpy(
            (void*)fluid.h_particles,
            (void*)optix_state.fluids.d_particles,
            fluid.count,
            cudaMemcpyDeviceToHost));
        return fluid;
    }

    float FluidModel::getTimeStep() { return fluid.dt; }

    void FluidModel::createOptixContext()
    {
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));

        auto context_log_cb = [](uint32_t level, const char* tag, const char* message, void* /*cbdata */) {
            fprintf(stderr, "[%2d][%12s]: %s\n", level, tag, message);
        };

        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = context_log_cb;
        options.logCallbackLevel = 4;
#ifndef NDEBUG
        // This may incur significant performance cost and should only be done during development.
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optix_state.context));
    }

    void FluidModel::prepareDataForOptix()
    {
        // copy solid boundaries to device
        size_t vertices_size_in_bytes = solid_boundary.vertices.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc((void**)&optix_state.boundary.d_vertex_buf, vertices_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(
            (void*)optix_state.boundary.d_vertex_buf,
            solid_boundary.vertices.data(), vertices_size_in_bytes,
            cudaMemcpyHostToDevice));

        size_t indices_size_in_bytes = solid_boundary.indices.size() * sizeof(uint3);
        CUDA_CHECK(cudaMalloc((void**)&optix_state.boundary.d_index_buf, indices_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(
            (void*)optix_state.boundary.d_index_buf,
            solid_boundary.indices.data(), indices_size_in_bytes,
            cudaMemcpyHostToDevice));

        size_t normals_size_in_bytes = solid_boundary.normals.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc((void**)&optix_state.boundary.d_normal_buf, normals_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(
            (void*)optix_state.boundary.d_normal_buf,
            solid_boundary.normals.data(), normals_size_in_bytes,
            cudaMemcpyHostToDevice));

        optix_state.boundary.num_primitive = solid_boundary.num_primitive;

        //copy fluid particles to device
        CUDA_CHECK(cudaMalloc((void**)&optix_state.fluids.d_particles, fluid.count));
        CUDA_CHECK(cudaMemcpy(
            (void*)optix_state.fluids.d_particles,
            (void*)fluid.h_particles,
            fluid.count,
            cudaMemcpyHostToDevice));

        uint32_t num_neighors = fluid.size * optix_state.config.num_neighbor;
        uint32_t neighors_id_in_bytes = num_neighors * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc((void**)&optix_state.fluids.d_neighbor_id, neighors_id_in_bytes));

        optix_state.fluids.num_particle = fluid.size;
        optix_state.fluids.density0 = fluid.density0;
        optix_state.fluids.mass0 = fluid.mass0;
        optix_state.fluids.radius = model_params.particle_radius;

        /*malloc device memory for Morton Code*/
        CUDA_CHECK(cudaMalloc((void**)&optix_state.d_Morton_code, fluid.size * sizeof(uint32_t)));


        CUDA_CHECK(cudaMalloc((void**)&optix_state.d_volume_rand_states, fluid.size * sizeof(RandState)));
        initRandomStates(optix_state.d_volume_rand_states, fluid.size);

        /*copy vpl random states to device*/
        size_t num_path_samples = optix_state.config.num_path_sample;
        CUDA_CHECK(cudaMalloc((void**)&optix_state.d_vpl_rand_states, num_path_samples * sizeof(RandState)));
        initRandomStates(optix_state.d_vpl_rand_states, num_path_samples);

        /*copy vpl records to device*/
        size_t num_vpl_data = optix_state.config.num_path_sample * optix_state.config.path_length;
        CUDA_CHECK(cudaMalloc((void**)&optix_state.d_vpl_records, num_vpl_data * sizeof(VPLRecord)));
    }

    void FluidModel::initLaunchParams()
    {
        optix_state.params.acc_structure = optix_state.scene_ias.handle;
        optix_state.params.boundary = optix_state.boundary;
        optix_state.params.config = optix_state.config;
        optix_state.params.fluids = optix_state.fluids;
        optix_state.params.wcoeff = wcoeff;
        optix_state.params.vpl_rand_states = optix_state.d_vpl_rand_states;
        optix_state.params.vpl_data = optix_state.d_vpl_records;

        CUDA_CHECK(cudaMalloc((void**)&optix_state.d_params, sizeof(Params)));
        CUDA_CHECK(cudaMemcpy(
            (void*)optix_state.d_params,
            (void*)&optix_state.params,
            sizeof(Params),
            cudaMemcpyHostToDevice));
    }

    void FluidModel::buildBoundaryGAS()
    {
        /* build solid boundary GAS*/
        uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
        triangle_input.triangleArray.numVertices = solid_boundary.vertices.size();
        triangle_input.triangleArray.vertexBuffers = (CUdeviceptr*)&optix_state.boundary.d_vertex_buf;
        triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.indexStrideInBytes = sizeof(uint3);
        triangle_input.triangleArray.numIndexTriplets = solid_boundary.indices.size();
        triangle_input.triangleArray.indexBuffer = (CUdeviceptr)optix_state.boundary.d_index_buf;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_state.context,
            &accel_options,
            &triangle_input,
            1,  // num_build_inputs
            &gas_buffer_sizes));

        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gas_buffer_sizes.tempSizeInBytes));

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = round(gas_buffer_sizes.outputSizeInBytes, 8ull);
        CUDA_CHECK(cudaMalloc(
            (void**)&d_buffer_temp_output_gas_and_compacted_size,
            compactedSizeOffset + 8));

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

        OPTIX_CHECK(optixAccelBuild(
            optix_state.context,
            0,                                  // CUDA stream
            &accel_options,
            &triangle_input,
            1,                                  // num build inputs
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size,
            gas_buffer_sizes.outputSizeInBytes,
            &optix_state.boundary_gas.handle,
            &emitProperty,                      // emitted property list
            1));                                // num emitted properties


        CUDA_CHECK(cudaFree((void*)d_temp_buffer));

        size_t compacted_gas_size;
        CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

        if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
        {
            CUDA_CHECK(cudaMalloc((void**)&optix_state.boundary_gas.d_output_buffer, compacted_gas_size));

            // use handle as input and output
            OPTIX_CHECK(optixAccelCompact(
                optix_state.context, 0, optix_state.boundary_gas.handle,
                optix_state.boundary_gas.d_output_buffer, compacted_gas_size,
                &optix_state.boundary_gas.handle));

            CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
        }
        else
        {
            optix_state.boundary_gas.d_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }

    void FluidModel::genParticleAABB()
    {
        const Particle* d_particles = optix_state.fluids.d_particles;
        float h = wcoeff.h;
        OptixAabb* d_aabbs = optix_state.fluids_gas.d_aabb_buffer;

        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(fluid.size),
            [=]__device__(uint32_t idx) {
                float3 center = d_particles[idx].pos;

                float3 min = center - h;
                float3 max = center + h;

                d_aabbs[idx] = {
                    min.x, min.y, min.z,
                    max.x, max.y, max.z
                };
            }
        );
    }

    void FluidModel::buildFluidGAS()
    {
        /* generate particles' AABBs */
        uint32_t num_prims = fluid.size;
        CUDA_CHECK(cudaMalloc((void**)&optix_state.fluids_gas.d_aabb_buffer, num_prims * sizeof(OptixAabb)));

        genParticleAABB();

        optix_state.fluids_gas.num_primitives = num_prims;

        /* build particles gas*/
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixBuildInput& custom_input = optix_state.fluids_gas.build_input;
        custom_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        custom_input.customPrimitiveArray.aabbBuffers = (CUdeviceptr*)&optix_state.fluids_gas.d_aabb_buffer;
        custom_input.customPrimitiveArray.numPrimitives = num_prims;
        custom_input.customPrimitiveArray.flags = optix_state.fluids_gas.custom_input_flags;
        custom_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes& gas_buffer_sizes = optix_state.fluids_gas.buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_state.context, &accel_options, &custom_input, 1, &gas_buffer_sizes));

        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer, gas_buffer_sizes.tempSizeInBytes));

        CUDA_CHECK(cudaMalloc(
            (void**)&optix_state.fluids_gas.d_output_buffer,
            gas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            optix_state.context,
            0,  // CUDA stream
            &accel_options,
            &custom_input,
            1,  // num build inputs
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            optix_state.fluids_gas.d_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &optix_state.fluids_gas.handle,
            nullptr,  // emitted property list
            0));      // num emitted properties

        CUDA_CHECK(cudaFree((void*)d_temp_buffer));

        /*malloc update buffer*/
        CUDA_CHECK(cudaMalloc(
            (void**)&optix_state.fluids_gas.d_update_buffer,
            optix_state.fluids_gas.buffer_sizes.tempUpdateSizeInBytes));
    }

    void FluidModel::updateFluidGAS()
    {
        /*update aabb buffer after one step*/
        genParticleAABB();

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

        OPTIX_CHECK(optixAccelBuild(
            optix_state.context,
            0,                       // CUDA stream
            &accel_options,
            &optix_state.fluids_gas.build_input,
            1,                                  // num build inputs
            optix_state.fluids_gas.d_update_buffer,
            optix_state.fluids_gas.buffer_sizes.tempUpdateSizeInBytes,
            optix_state.fluids_gas.d_output_buffer,
            optix_state.fluids_gas.buffer_sizes.outputSizeInBytes,
            &optix_state.fluids_gas.handle,
            nullptr,                           // emitted property list
            0));                               // num emitted properties
    }

    void FluidModel::buildSceneIAS()
    {
        sortParticlesByMortonCode();
        buildBoundaryGAS();
        buildFluidGAS();

        float transform[12] = { 1.f, 0.f, 0.f, 0.f ,0.f, 1.f, 0.f, 0.f ,0.f, 0.f, 1.f, 0.f };

        OptixInstance instances[2] = {};

        instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[0].instanceId = 0;
        instances[0].sbtOffset = 0;
        instances[0].visibilityMask = OBJECT_SOLID;
        instances[0].traversableHandle = optix_state.boundary_gas.handle;
        memcpy(&instances[0].transform, transform, 12 * sizeof(float));

        instances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[1].instanceId = 1;
        instances[1].sbtOffset = 0;
        instances[1].visibilityMask = OBJECT_FLUID;
        instances[1].traversableHandle = optix_state.fluids_gas.handle;
        memcpy(&instances[1].transform, transform, 12 * sizeof(float));

        CUdeviceptr& d_instance = optix_state.scene_ias.d_instances;
        size_t      instances_size_in_bytes = 2 * sizeof(OptixInstance);
        CUDA_CHECK(cudaMalloc((void**)&optix_state.scene_ias.d_instances, instances_size_in_bytes));
        CUDA_CHECK(cudaMemcpy((void*)d_instance, instances, instances_size_in_bytes, cudaMemcpyHostToDevice));

        OptixBuildInput& instance_input = optix_state.scene_ias.build_input;
        instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances = d_instance;
        instance_input.instanceArray.numInstances = 2;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes& ias_buffer_sizes = optix_state.scene_ias.buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_state.context, &accel_options, &instance_input, 1, &ias_buffer_sizes));
        CUdeviceptr d_temp_buffer_ias;
        CUDA_CHECK(cudaMalloc((void**)&d_temp_buffer_ias, ias_buffer_sizes.tempSizeInBytes));

        CUDA_CHECK(cudaMalloc((void**)&optix_state.scene_ias.d_output_buffer, ias_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            optix_state.context,
            0,  // CUDA stream
            &accel_options,
            &instance_input,
            1,  // num build inputs
            d_temp_buffer_ias,
            ias_buffer_sizes.tempSizeInBytes,
            optix_state.scene_ias.d_output_buffer,
            ias_buffer_sizes.outputSizeInBytes,
            &optix_state.scene_ias.handle,
            nullptr,
            0));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_ias)));

        /*malloc update buffer*/
        CUDA_CHECK(cudaMalloc(
            (void**)&optix_state.scene_ias.d_update_buffer,
            optix_state.scene_ias.buffer_sizes.tempUpdateSizeInBytes));
    }

    void FluidModel::updateSceneIAS()
    {
        sortParticlesByMortonCode();

        updateFluidGAS();

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

        OPTIX_CHECK(optixAccelBuild(
            optix_state.context,
            0,            // CUDA stream
            &accel_options,
            &optix_state.scene_ias.build_input,
            1,                  // num build inputs
            optix_state.scene_ias.d_update_buffer,
            optix_state.scene_ias.buffer_sizes.tempUpdateSizeInBytes,
            optix_state.scene_ias.d_output_buffer,
            optix_state.scene_ias.buffer_sizes.outputSizeInBytes,
            &optix_state.scene_ias.handle,
            nullptr,            // emitted property list
            0));                  // num emitted properties  
    }

    void FluidModel::createMoudle()
    {
        optix_state.pipeline_compile_options.usesMotionBlur = false;
        optix_state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        optix_state.pipeline_compile_options.numPayloadValues = 0;
        optix_state.pipeline_compile_options.numAttributeValues = 0;
        optix_state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        optix_state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        std::vector<OptixPayloadType> pay_load_types;
        {
            OptixPayloadType pay_load_type;
            pay_load_type.numPayloadValues = sizeof(vplPointPayloadSemantics) / sizeof(vplPointPayloadSemantics[0]);
            pay_load_type.payloadSemantics = vplPointPayloadSemantics;
            pay_load_types.push_back(pay_load_type);
        }

        {
            OptixPayloadType pay_load_type;
            pay_load_type.numPayloadValues = sizeof(boundaryValuePRDSemantics) / sizeof(boundaryValuePRDSemantics[0]);
            pay_load_type.payloadSemantics = boundaryValuePRDSemantics;
            pay_load_types.push_back(pay_load_type);
        }

        {
            OptixPayloadType pay_load_type;
            pay_load_type.numPayloadValues = sizeof(neighborSearchPRDSemantics) / sizeof(neighborSearchPRDSemantics[0]);
            pay_load_type.payloadSemantics = neighborSearchPRDSemantics;
            pay_load_types.push_back(pay_load_type);
        }

        OptixModuleCompileOptions module_compile_options = {};
#ifndef NDEBUG
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
        module_compile_options.numPayloadTypes = pay_load_types.size();
        module_compile_options.payloadTypes = pay_load_types.data();

        OPTIX_CHECK_LOG(optixModuleCreate(
            optix_state.context,
            &module_compile_options,
            &optix_state.pipeline_compile_options,
            FluidSolver_ptx,
            strlen(FluidSolver_ptx),
            LOG, &LOG_SIZE,
            &optix_state.fluid_solver_module));
    }

    void FluidModel::createProgramGroups()
    {
        OptixProgramGroupOptions program_group_options = {};

        // ray generation program groups
        {
            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = optix_state.fluid_solver_module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__project_vpl_construct";

            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                optix_state.context,
                &raygen_prog_group_desc,
                1,                             
                &program_group_options,
                LOG, &LOG_SIZE,
                &optix_state.raygen_press_proj_construct));

            memset(&raygen_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = optix_state.fluid_solver_module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__neighbor_search";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                optix_state.context,
                &raygen_prog_group_desc,
                1,                             
                &program_group_options,
                LOG, &LOG_SIZE,
                &optix_state.raygen_neighbor_search));

        }

        // hit program groups
        {
            OptixProgramGroupDesc hit_prog_group_desc = {};

            memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH = optix_state.fluid_solver_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__sample_vpl_source_point";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                optix_state.context,
                &hit_prog_group_desc,
                1,                             
                &program_group_options,
                LOG, &LOG_SIZE,
                &optix_state.hit_sample_vpl_source_point));
            
            memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH = optix_state.fluid_solver_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__sample_vpl_point";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                optix_state.context,
                &hit_prog_group_desc,
                1,                             
                &program_group_options,
                LOG, &LOG_SIZE,
                &optix_state.hit_sample_vpl_point));

            memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleIS = optix_state.fluid_solver_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__approx_boundary_value";
            hit_prog_group_desc.hitgroup.moduleAH = optix_state.fluid_solver_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__approx_boundary_value";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                optix_state.context,
                &hit_prog_group_desc,
                1,                             
                &program_group_options,
                LOG, &LOG_SIZE,
                &optix_state.hit_approx_boundary_value));

            memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleIS = optix_state.fluid_solver_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__neighbor_search";
            hit_prog_group_desc.hitgroup.moduleAH = optix_state.fluid_solver_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__neighbor_search";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                optix_state.context,
                &hit_prog_group_desc,
                1,                            
                &program_group_options,
                LOG, &LOG_SIZE,
                &optix_state.hit_neighbor_search));
        }

        // miss program group
        {
            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                optix_state.context,
                &miss_prog_group_desc,
                1,
                &program_group_options,
                LOG, &LOG_SIZE,
                &optix_state.miss_default));
        }
    }

    void FluidModel::createSBT()
    {
        // sbt record of ray generation
        const size_t raygen_record_size = sizeof(EmptyRecord);

        {
            EmptyRecord raygen_sbt = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(optix_state.raygen_press_proj_construct, &raygen_sbt));

            CUDA_CHECK(cudaMalloc((void**)&optix_state.d_raygen_proj_construct_record, raygen_record_size));
            CUDA_CHECK(cudaMemcpy(
                (void*)optix_state.d_raygen_proj_construct_record,
                &raygen_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice));
        }

        {
            EmptyRecord raygen_sbt = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(optix_state.raygen_neighbor_search, &raygen_sbt));

            CUDA_CHECK(cudaMalloc((void**)&optix_state.d_raygen_neighbor_search_record, raygen_record_size));
            CUDA_CHECK(cudaMemcpy(
                (void*)optix_state.d_raygen_neighbor_search_record,
                &raygen_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice));
        }

        // sbt record of hit group
        size_t hitgroup_record_size = sizeof(EmptyRecord);

        {
            EmptyRecord hitgroup_records[3] = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(optix_state.hit_sample_vpl_point, &hitgroup_records[0]));
            OPTIX_CHECK(optixSbtRecordPackHeader(optix_state.hit_approx_boundary_value, &hitgroup_records[1]));
            OPTIX_CHECK(optixSbtRecordPackHeader(optix_state.hit_sample_vpl_source_point, &hitgroup_records[2]));

            CUDA_CHECK(cudaMalloc((void**)&optix_state.d_hit_project_records, 3 * hitgroup_record_size));
            CUDA_CHECK(cudaMemcpy(
                (void*)optix_state.d_hit_project_records,
                hitgroup_records,
                3 * hitgroup_record_size,
                cudaMemcpyHostToDevice));
        }

        {
            EmptyRecord hitgroup_records = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(optix_state.hit_neighbor_search, &hitgroup_records));

            CUDA_CHECK(cudaMalloc((void**)&optix_state.d_hit_neighbor_search_record, hitgroup_record_size));
            CUDA_CHECK(cudaMemcpy(
                (void*)optix_state.d_hit_neighbor_search_record,
                &hitgroup_records,
                hitgroup_record_size,
                cudaMemcpyHostToDevice));
        }

        // sbt record of miss group
        size_t missgroup_record_size = sizeof(EmptyRecord);
        {
            EmptyRecord missgroup_records = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(optix_state.miss_default, &missgroup_records));

            CUDA_CHECK(cudaMalloc((void**)&optix_state.d_missgroup_records, missgroup_record_size));
            CUDA_CHECK(cudaMemcpy(
                (void*)optix_state.d_missgroup_records,
                &missgroup_records,
                missgroup_record_size,
                cudaMemcpyHostToDevice));
        }

        optix_state.sbt_press_proj_construct.raygenRecord = optix_state.d_raygen_proj_construct_record;
        optix_state.sbt_press_proj_construct.hitgroupRecordBase = optix_state.d_hit_project_records;
        optix_state.sbt_press_proj_construct.hitgroupRecordCount = 3;
        optix_state.sbt_press_proj_construct.hitgroupRecordStrideInBytes = hitgroup_record_size;
        optix_state.sbt_press_proj_construct.missRecordBase = optix_state.d_missgroup_records;
        optix_state.sbt_press_proj_construct.missRecordCount = 1;
        optix_state.sbt_press_proj_construct.missRecordStrideInBytes = missgroup_record_size;

        optix_state.sbt_neighbor_search.raygenRecord = optix_state.d_raygen_neighbor_search_record;
        optix_state.sbt_neighbor_search.hitgroupRecordBase = optix_state.d_hit_neighbor_search_record;
        optix_state.sbt_neighbor_search.hitgroupRecordCount = 1;
        optix_state.sbt_neighbor_search.hitgroupRecordStrideInBytes = hitgroup_record_size;
        optix_state.sbt_neighbor_search.missRecordBase = optix_state.d_missgroup_records;
        optix_state.sbt_neighbor_search.missRecordCount = 1;
        optix_state.sbt_neighbor_search.missRecordStrideInBytes = missgroup_record_size;

    }

    void FluidModel::createPipeline()
    {
        uint32_t max_trace_depth = 4;
        uint32_t max_cc_depth = 0;
        uint32_t max_dc_depth = 4;
        uint32_t max_traversal_depth = 2;
        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;

        // create projection vpl construction pipeline
        {
            std::vector<OptixProgramGroup> program_groups;
            program_groups.push_back(optix_state.raygen_press_proj_construct);
            program_groups.push_back(optix_state.hit_sample_vpl_point);
            program_groups.push_back(optix_state.hit_approx_boundary_value);
            program_groups.push_back(optix_state.hit_sample_vpl_source_point);
            program_groups.push_back(optix_state.miss_default);

            OPTIX_CHECK_LOG(optixPipelineCreate(
                optix_state.context,
                &optix_state.pipeline_compile_options,
                &pipeline_link_options,
                program_groups.data(), // ptr to first program group
                program_groups.size(), // number of program groups
                LOG, &LOG_SIZE,
                &optix_state.press_proj_construct_pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups)
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, optix_state.press_proj_construct_pipeline));

            OPTIX_CHECK(optixUtilComputeStackSizes(
                &stack_sizes,
                max_trace_depth,
                max_cc_depth,  // maxCCDepth
                max_dc_depth,  // maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(
                optix_state.press_proj_construct_pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversal_depth));  // maxTraversableDepth
        }

        // create neighbor search pipeline 
        {
            std::vector<OptixProgramGroup> program_groups;

            program_groups.push_back(optix_state.raygen_neighbor_search);
            program_groups.push_back(optix_state.hit_neighbor_search);
            program_groups.push_back(optix_state.miss_default);

            OPTIX_CHECK_LOG(optixPipelineCreate(
                optix_state.context,
                &optix_state.pipeline_compile_options,
                &pipeline_link_options,
                program_groups.data(), // ptr to first program group
                program_groups.size(), // number of program groups
                LOG, &LOG_SIZE,
                &optix_state.neighbor_search_pipeline));
            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups)
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, optix_state.neighbor_search_pipeline));
            OPTIX_CHECK(optixUtilComputeStackSizes(
                &stack_sizes,
                max_trace_depth,
                max_cc_depth,  // maxCCDepth
                max_dc_depth,  // maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(
                optix_state.neighbor_search_pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversal_depth));  // maxTraversableDepth
        }
    }

    void FluidModel::neighborSearch()
    {
        OPTIX_CHECK(optixLaunch(
            optix_state.neighbor_search_pipeline,
            0,
            optix_state.d_params,
            sizeof(Params),
            &optix_state.sbt_neighbor_search,
            256,
            256,
            (fluid.size + 256 * 256) / (256 * 256)));
    }

    void FluidModel::calcDensity()
    {
        Particle* d_particles = optix_state.fluids.d_particles;
        uint32_t* d_neighbors = optix_state.fluids.d_neighbor_id;
        uint32_t num_neighbor = optix_state.config.num_neighbor;

        SPHKernelWCoeff wcoeff = this->wcoeff;

        float mj = fluid.mass0;
        float density0 = fluid.density0;
        
        auto calc_density = [=] __device__(uint32_t idx) {
            Particle& pi = d_particles[idx];
            float density = 0.f;
            for (uint32_t i = 0; i < pi.num_neighbor; i++)
            {
                const Particle& pj = d_particles[d_neighbors[idx * num_neighbor + i]];
                float Wij = W(pi.pos - pj.pos, wcoeff);
                density += mj * Wij;
            }
            pi.rho = density < 1e-9f ? density0 : density;
        };

        /*update the density*/
        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(fluid.size),
            calc_density);
    }

    void FluidModel::calcNonPressure()
    {
        Particle* d_particles = optix_state.fluids.d_particles;
        uint32_t* d_neighbors = optix_state.fluids.d_neighbor_id;
        uint32_t num_neighbors = optix_state.config.num_neighbor;

        float3 g = model_params.gravity;

        float mj = fluid.mass0;
        SPHKernelDWCoeff dwcoeff = this->dwcoeff;

        float viscos_coeff = model_params.viscos_coeff;

        auto calc_non_pressure = [=] __device__(uint32_t idx) {
            Particle& pi = d_particles[idx];
            float3 viscos = make_float3(0.f);
            float h = dwcoeff.h;
            for (uint32_t i = 0; i < pi.num_neighbor; i++)
            {
                const Particle& pj = d_particles[d_neighbors[idx * num_neighbors + i]];
                float rhoj = pj.rho;
                float3 xij = pi.pos - pj.pos;
                float3 vij = pi.vel - pj.vel;
                float3 dWij = dW(xij, dwcoeff);

                viscos += mj / rhoj * dot(vij, xij) / (length2(xij) + 0.01f * h * h) * dWij;
            }

            pi.acc = g + 10.f * viscos_coeff * viscos;
        };

        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(fluid.size),
            calc_non_pressure);
    }

    void FluidModel::computeTimeStep()
    {
        Particle* d_particles = optix_state.fluids.d_particles;
        uint32_t num_particle = optix_state.fluids.num_particle;

        float h = fluid.dt;

        thrust::device_ptr<Particle> d_particle_ptr(d_particles);
        auto p = thrust::max_element(d_particle_ptr, d_particle_ptr + num_particle,
            [=]__device__(const Particle & a, const Particle & b) {
            return length2(a.vel + a.acc * h) < length2(b.vel + b.acc * h);
        });

        Particle particle;
        CUDA_CHECK(cudaMemcpy(
            (void*)&particle,
            (void*)p.get(),
            sizeof(Particle),
            cudaMemcpyDeviceToHost));

        float max_vel = fmaxf(length(particle.vel + particle.acc * h), 1e-9f);
        float cfl_dt = model_params.cfl_factor * 0.4f * 2.f * model_params.particle_radius / max_vel;

        cfl_dt = clamp(cfl_dt, model_params.min_cfl_dt, model_params.max_cfl_dt);
        fluid.dt = cfl_dt;
    }

    void FluidModel::predictVel()
    {
        Particle* d_particles = optix_state.fluids.d_particles;
        float t = fluid.dt;

        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(fluid.size),
            [=] __device__(uint32_t idx) {
            d_particles[idx].vel += t * d_particles[idx].acc;
        });
    }

    void FluidModel::correctVel()
    {
        calcDivAndNormal();
        START_TIMING("compute volume integral term");
            calcVolumeIntegralTerm();
        STOP_TIMING_AVG;
        calcBoundaryIntegralTerm();

        Particle* d_particles = optix_state.fluids.d_particles;

        auto pressure_projection = [=]__device__(uint32_t idx) {
            Particle& particle = d_particles[idx];

            float3 volume_term = -particle.volume_term;
            float3 boundary_term = particle.boundary_term;

            particle.vel -= volume_term + boundary_term;

            float v_dot_n = dot(particle.vel, particle.bNormal);
            if (v_dot_n > 0.f)
                particle.vel -= v_dot_n * particle.bNormal;
        };

        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(fluid.size),
            pressure_projection);
    }

    void FluidModel::calcDivAndNormal()
    {
        Particle* d_particles = optix_state.fluids.d_particles;
        uint32_t* d_neighbors = optix_state.fluids.d_neighbor_id;
        uint32_t num_neighbors = optix_state.config.num_neighbor;

        float mj = fluid.mass0;
        SPHKernelDWCoeff dwcoeff = this->dwcoeff;

        auto calc_div_and_normal = [=] __device__(uint32_t idx) {
            Particle& pi = d_particles[idx];
            float div = 0.f;
            float3 normal = make_float3(0.f);

            for (uint32_t j = 0; j < pi.num_neighbor; j++)
            {
                const Particle& pj = d_particles[d_neighbors[idx * num_neighbors + j]];
                float rhoj = pj.rho;
                float3 xij = pi.pos - pj.pos;
                float3 vij = pi.vel - pj.vel;
                float3 dWij = dW(xij, dwcoeff);
                div += mj / rhoj * dot(vij, dWij);

                normal += mj / rhoj * dWij;
            }

            pi.div = -div;
            pi.normal = -normal;
        };

        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(fluid.size),
            calc_div_and_normal);
    }

    void FluidModel::calcVolumeIntegralTerm()
    {
        Particle* d_particles = optix_state.fluids.d_particles;
		uint32_t num_particle =  optix_state.fluids.num_particle; 

        float eps = model_params.dGdx_reg;
        float V0 = fluid.mass0 / fluid.density0;

        RandState* d_rand_state = optix_state.d_volume_rand_states;
        uint32_t n = model_params.num_volume_sample; 
        float inv_sample_num = 1.f / n;
        float c = model_params.volume_coeff;
        float inv_pdf = num_particle * c * V0;

        auto uniform_sampling = [=] __device__(uint32_t idx) {
			Particle& pi = d_particles[idx];
            float3 xi = pi.pos;
            KahanSum volume_term;
            for (uint32_t j = 0; j < n; j++)
            {
                uint32_t rand_idx = (uint32_t)(rand_uniform(d_rand_state[idx]) * (num_particle - 1));
                if(rand_idx == idx) continue;
                const Particle& pj = d_particles[rand_idx];
                float3 xj = pj.pos;
                float divj = pj.div;

                float3 dGdx = Poisson_dGdx(xi, xj, eps);
                volume_term += dGdx * divj;
            }
            pi.volume_term = inv_sample_num * inv_pdf * volume_term.sum;
        };

        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(num_particle),
            uniform_sampling);
    }

    void FluidModel::calcBoundaryIntegralTerm()
    {
        uint32_t num_path_sample = optix_state.config.num_path_sample;
        uint32_t num_records = num_path_sample * optix_state.config.path_length;

        VPLRecord* d_vpl_records = optix_state.d_vpl_records;
        CUDA_CHECK(cudaMemset(d_vpl_records, 0, num_records * sizeof(VPLRecord)));

        OPTIX_CHECK(optixLaunch(
            optix_state.press_proj_construct_pipeline,
            0,
            optix_state.d_params,
            sizeof(Params),
            &optix_state.sbt_press_proj_construct,
            256,
            256,
            (num_path_sample + 256 * 256) / (256 * 256)));

        thrust::device_ptr<VPLRecord> begin(d_vpl_records);
        auto end = thrust::remove_if(begin, begin + num_records, []__device__(const VPLRecord & record)->bool { return record.value == 0.f; });
        uint32_t num_valid_record = end - begin;
        fluid.num_vpl = num_valid_record;

        Particle* d_particles = optix_state.fluids.d_particles;
        float eps = model_params.dGdx_reg;
        
        auto calc_boundary_integral_term = [=]__device__(uint32_t idx) {
            Particle& particle = d_particles[idx];

            KahanSum sum;
            for (uint32_t i = 0; i < num_valid_record; i++)
            {
                float3 dGdx = Poisson_dGdx(particle.pos, d_vpl_records[i].pos, eps);
                sum += d_vpl_records[i].value * dGdx;
            }
            particle.boundary_term = sum.sum;
        };

        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(fluid.size),
            calc_boundary_integral_term);
    }

    void FluidModel::updatePos()
    {
        Particle* d_particles = optix_state.fluids.d_particles;
        float t = fluid.dt;

        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(fluid.size),
            [=] __device__(uint32_t idx) {
                d_particles[idx].pos += d_particles[idx].vel * t;
        });
    }

    void FluidModel::sortParticlesByMortonCode()
    {
        uint32_t* d_Morton_code = optix_state.d_Morton_code;
        Particle* d_particles = optix_state.fluids.d_particles;
        thrust::for_each(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(fluid.size),
            [=] __device__(uint32_t idx) {
            float3 pos = d_particles[idx].pos;
            d_Morton_code[idx] = MortonCode3(__float_as_uint(pos.x), __float_as_uint(pos.y), __float_as_uint(pos.z));
        }
        );

        thrust::device_ptr<uint32_t> d_keys(d_Morton_code);
        thrust::device_ptr<Particle> d_values(d_particles);
        thrust::sort_by_key(d_keys, d_keys + fluid.size, d_values);
    }
}
