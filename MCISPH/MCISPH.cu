#include"MCISPH.h"
#include"Utils/json.hpp"

#include "Utils/Exception.h"
#include "Utils/Timing.h"

#include <fstream>
#include <Partio.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "Utils/tiny_obj_loader.h"

const std::string getConfigPath()
{
#ifdef CONFIG_PATH
    return CONFIG_PATH;
#elif
    return "../configs/";
#endif
}

namespace mcisph
{
    void MCISPH::loadConfig(std::string json)
    {
        nlohmann::json json_config;
        {
            std::ifstream config_file(json.c_str());
            if (!config_file)
            {
                fprintf(stderr, "Failed to load SPH config file: %s", json.c_str());
                exit(1);
            }
            config_file >> json_config;
            config_file.close();
        }

        output_dir = getConfigPath() + json_config["output_dir"].get<std::string>();
        bgeo_file = getConfigPath()+json_config["bgeo_file"].get<std::string>();
        obj_file = getConfigPath()+json_config["obj_file"].get<std::string>();
        
        max_time = json_config["time"].get<float>();
        frame_dt = 1.f / json_config["frame_rate"].get<float>();
            
        FluidModelConfig host_config = {};
        
        if(json_config.contains("viscosity_coeff"))
            host_config.viscos_coeff = json_config["viscosity_coeff"].get<float>();
        else
            host_config.viscos_coeff = 0.0f;

        host_config.particle_radius = json_config["particle_radius"].get<float>();
        host_config.density0 = json_config["density"].get<float>();

        host_config.gravity.x = json_config["gravity"][0].get<float>();
        host_config.gravity.y = json_config["gravity"][1].get<float>();
        host_config.gravity.z = json_config["gravity"][2].get<float>();

        host_config.cfl_factor = json_config["cfl_factor"].get<float>();
		host_config.min_cfl_dt = json_config["min_cfl_dt"].get<float>();
		host_config.max_cfl_dt = json_config["max_cfl_dt"].get<float>();
        
        host_config.dGdx_reg = powf(2.f * host_config.particle_radius, 2.0f);

        host_config.num_volume_sample = json_config["num_volume_sample"].get<uint32_t>();
        host_config.volume_coeff = json_config["volume_coeff"].get<float>();

        DeviceConfig device_config = {};

        device_config.num_neighbor = json_config["num_neighbor"].get<uint32_t>();
        device_config.path_length = json_config["path_length"].get<uint32_t>() + 1;
        device_config.area_coeff = json_config["area_coeff"].get<float>();
        
        fluid_model.setConfig(std::move(host_config), std::move(device_config));
    }

    void MCISPH::initFluidModel()
    {
        loadFluid();
        loadBoundary();

        fluid_model.setDevice();
        fluid_model.createOptixContext();
        fluid_model.prepareDataForOptix();
        fluid_model.buildSceneIAS();
        fluid_model.createMoudle();
        fluid_model.createProgramGroups();
        fluid_model.createPipeline();
        fluid_model.createSBT();
        fluid_model.initLaunchParams();
    }

    void MCISPH::loadFluid()
    {
        /*load particle data to CPU memory*/
        Partio::ParticlesDataMutable* data = Partio::read(bgeo_file.c_str());
        Partio::ParticleAttribute pos_attr;
        data->attributeInfo("position", pos_attr);
        Partio::ParticleAttribute vel_attr;
        bool has_vel = data->attributeInfo("velocity", vel_attr);

        int count = data->numParticles();
        std::vector<Particle> particles(count);

        for (int i = 0; i < count; i++)
        {
            const float* pos = data->data<float>(pos_attr, i);
            particles[i].pos = { pos[0], pos[1], pos[2] };
            if (has_vel)
            {
                const float* vel = data->data<float>(vel_attr, i);
                particles[i].vel = { vel[0], vel[1], vel[2] };
            }
            else
            {
                particles[i].vel = { 0.f, 0.f, 0.f };
            }
        }

        data->release();

        printf("Particle counts: %d\n", count);
        
        fluid_model.setFluidParticles(particles);
    }

    void MCISPH::loadBoundary()
    {
        tinyobj::ObjReaderConfig reader_config;
        tinyobj::ObjReader reader;

        /*load boundary meshes to CPU memory*/
        if (!reader.ParseFromFile(obj_file, reader_config))
        {
            if (!reader.Error().empty()) 
            { 
                fprintf(stderr, "TinyObjReader:%s\n", reader.Error().c_str()); 
                exit(1);
            }
        }
        if (!reader.Warning().empty()) printf("TinyObjReader:%s\n", reader.Warning().c_str());

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();

        Boundary boundary;
        boundary.vertices.resize(attrib.vertices.size() / 3);

        for (size_t v = 0; v < attrib.vertices.size() / 3; v++)
            boundary.vertices[v] = { attrib.vertices[3 * v],attrib.vertices[3 * v + 1],attrib.vertices[3 * v + 2] };


        for (size_t s = 0; s < shapes.size(); s++)
        {
            boundary.indices.reserve(boundary.indices.size() + shapes[s].mesh.num_face_vertices.size());
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
            {
                uint3 indice;
                indice.x = shapes[s].mesh.indices[3 * f].vertex_index;
                indice.y = shapes[s].mesh.indices[3 * f + 1].vertex_index;
                indice.z = shapes[s].mesh.indices[3 * f + 2].vertex_index;
                boundary.indices.push_back(indice);
            }
        }

        size_t num_primitive = boundary.indices.size();
        boundary.num_primitive = num_primitive;
        boundary.normals.resize(num_primitive);

        for (size_t f = 0; f < num_primitive; f++)
        {
            float3 v0 = boundary.vertices[boundary.indices[f].x];
            float3 v1 = boundary.vertices[boundary.indices[f].y];
            float3 v2 = boundary.vertices[boundary.indices[f].z];
            float3 e0 = v1 - v0;
            float3 e1 = v2 - v0;
            float3 n = cross(e0, e1);
            boundary.normals[f] = normalize(n);
        }
        
        printf("Boundary vertices:%zu, primitives:%zu\n", boundary.vertices.size(), boundary.indices.size());

        fluid_model.setBoundary(std::move(boundary));
    }

    void MCISPH::simulate()
    {
        while(current_t < max_time)
        {
            START_TIMING("total time");

              START_TIMING("neighborhood search");
              fluid_model.neighborSearch();
              STOP_TIMING_AVG;

              START_TIMING("compute non-pressure foreces");
              fluid_model.calcDensity();
              fluid_model.calcNonPressure();
              STOP_TIMING_AVG;     
              
              fluid_model.computeTimeStep(); 
              fluid_model.predictVel();

              START_TIMING("pressure projection");
              fluid_model.correctVel(); 
              STOP_TIMING_AVG;

              fluid_model.updatePos();
              fluid_model.updateSceneIAS();

            STOP_TIMING_AVG;

            saveFluidState();
        }

        std::cout << std::endl << "-----------------------------------" << std::endl;
        PRINT_TIMING_AVG
        PRINT_TIMING_SUMS
    }

    void MCISPH::saveFluidState()
    {
        static uint64_t num_vpl = 0;
        static uint32_t frame_count = 0;

        ++time_steps;
        current_t += fluid_model.getTimeStep();
        if(current_t < frame_count * frame_dt) return;

        ++frame_count;

        const Fluid& fluid = fluid_model.getFluid();
        num_vpl += fluid.num_vpl;

        printf("\npresent time: %f\n", current_t);
        printf("average adaptive time: %lf\n", current_t / time_steps);
        printf("average number of VPLs:%lu\n", num_vpl / time_steps);

        uint32_t num_particles = fluid.size;
        Particle* particles = fluid.h_particles;

        Partio::ParticlesDataMutable& particleData = *Partio::create();
        Partio::ParticleAttribute posAttr = particleData.addAttribute("position", Partio::VECTOR, 3);
        Partio::ParticleAttribute velAttr = particleData.addAttribute("velocity", Partio::VECTOR, 3);

        for (uint32_t i = 0; i < num_particles; i++)
        {
            Partio::ParticleIndex index = particleData.addParticle();
            float* pos = particleData.dataWrite<float>(posAttr, index);
            float* vel = particleData.dataWrite<float>(velAttr, index);

            const Particle& particle = particles[i];
            pos[0] = particle.pos.x;
            pos[1] = particle.pos.y;
            pos[2] = particle.pos.z;

            vel[0] = particle.vel.x;
            vel[1] = particle.vel.y;
            vel[2] = particle.vel.z;
        }

        std::string output_file = output_dir + "FluidData" + std::to_string(frame_count) + ".bgeo";

        Partio::write(output_file.c_str(), particleData, true);
        particleData.release();
    }
}