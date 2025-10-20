#pragma once
#include "Common.cuh"

namespace mcisph
{
	enum ObjectType
	{
		OBJECT_FLUID = 1,
		OBJECT_SOLID = 1 << 1,
		OBJECT_ANY = 0xFF
	};

	/* Ray types*/
	enum RayType
	{
		RAY_TYPE_IMMEDIATE_POINT = 0,
		RAY_TYPE_BOUNDARY_VALUE,
		RAY_TYPE_SOURCE_POINT
	};

	struct Particle
	{
		float3 pos;
		float3 vel;
		float3 acc;
		float3 boundary_term;
		float3 volume_term;
		float3 bNormal;
		float3 normal;
		uint32_t num_neighbor;
		float rho;
		float div;
	};

	struct Boundary
	{
		std::vector<float3> vertices;
		std::vector<uint3> indices;
		std::vector<float3> normals;

		uint32_t num_primitive;
	};

	struct Fluid
	{
		Particle* h_particles; // pinned memory
		uint32_t size;		   // num particles
		uint32_t count;		   // total bytes
		uint32_t num_vpl;
		float dt;
		float mass0;
		float density0;
	};

	struct DeviceBoundary
	{
		float3* d_vertex_buf;
		uint3*	d_index_buf;
		float3* d_normal_buf;

		uint32_t num_primitive;
	};

	struct DeviceFluid
	{
		Particle* d_particles;
		uint32_t* d_neighbor_id;
		uint32_t num_particle;
		float radius;
		float mass0;
		float density0;
	};
}