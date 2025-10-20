#include "FluidSolver.h"
#include <optix_device.h>

using namespace mcisph;
extern "C" { __constant__ Params params; }

//==========================
// sample VPL points
//==========================

//sample the source point
__forceinline__ __device__ VPLSourcePoint sampleVPLSourcePoint(uint32_t idx)
{
	OptixTraversableHandle handle = params.acc_structure;
	const DeviceFluid& fluids = params.fluids;

	const Particle& particle = fluids.d_particles[idx];
	float3 pos = particle.pos;
	float radius = fluids.radius;
	float tmax = params.wcoeff.h;

	float3 sample_pos;
	uint32_t prim_id = -1;
	float factor = params.config.area_coeff * 4.f * radius * radius;

	uint32_t u[4] = {-1};

	float dis = tmax;
	auto sample = [&]__device__(float3 dir){
		optixTrace(
			PAYLOAD_TYPE_SOURCE_POINT,
			handle,
			pos, normalize(dir),
			1e-16f, tmax, 0.f,
			OBJECT_SOLID,
			OPTIX_RAY_FLAG_NONE,
			RAY_TYPE_SOURCE_POINT,
			0,
			0,
			u[0], u[1], u[2], u[3]);

		uint32_t id = u[0];	
		float3 point = make_float3(__uint_as_float(u[1]), __uint_as_float(u[2]), __uint_as_float(u[3]));
		
		float len = length(point - pos);
		if(dis > len && id != -1)
		{
			dis = len;
			sample_pos = point;
			prim_id = id;
		}
	};

	sample(particle.normal);
	sample(particle.acc);
	sample(particle.vel);

	return {prim_id, sample_pos, factor};
}

extern "C" __global__ void __closesthit__sample_vpl_source_point()
{
	optixSetPayloadTypes(PAYLOAD_TYPE_SOURCE_POINT);

	uint32_t prim_id = optixGetPrimitiveIndex();
	float3 ray_orgin = optixGetWorldRayOrigin();
	float3 ray_dir   = optixGetWorldRayDirection();
	float hit_t      = optixGetRayTmax();

	float3 hit_pos = ray_orgin + hit_t * ray_dir;

	optixSetPayload_0(prim_id);
	optixSetPayload_1(__float_as_uint(hit_pos.x));
	optixSetPayload_2(__float_as_uint(hit_pos.y));
	optixSetPayload_3(__float_as_uint(hit_pos.z));
}

//sample intermediate point
__forceinline__ __device__ float3 uniform_hemisphere_sample(const float3 n, RandState& rand_state)
{
	float phi = 2.f * M_PIf * rand_uniform(rand_state);
	float z = 2.f * rand_uniform(rand_state) - 1.f;
	float x = cosf(phi) * sqrtf(1.f - z * z);
	float y = sinf(phi) * sqrtf(1.f - z * z);
	float3 in_ball = { x, y, z };
	return dot(in_ball, n) < 0.f ? -in_ball : in_ball;
}

__forceinline__ __device__ VPLPoint sampleVPLPoint(const VPLPoint& point, RandState& rand_state)
{
	float3 n = params.boundary.d_normal_buf[point.prim_id];
	float3 dir = uniform_hemisphere_sample(n, rand_state);

	uint32_t u[4] = {-1};

	optixTrace(
		PAYLOAD_TYPE_IMMEDIATE_POINT,
		params.acc_structure,
		point.pos, dir,
		1e-16f, 1e16f, 0.0f,
		OBJECT_SOLID,  //the fluid should not see it
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_IMMEDIATE_POINT,
		0,
		0,
		u[0], u[1], u[2], u[3]);

	VPLPoint sample;
	sample.prim_id = u[0];
	sample.pos = { __uint_as_float(u[1]), __uint_as_float(u[2]), __uint_as_float(u[3]) };

	return sample;
}

extern "C" __global__ void __closesthit__sample_vpl_point()
{
	optixSetPayloadTypes(PAYLOAD_TYPE_IMMEDIATE_POINT);

	uint32_t prim_id = optixGetPrimitiveIndex();
	float3 ray_orgin = optixGetWorldRayOrigin();
	float3 ray_dir   = optixGetWorldRayDirection();
	float hit_t      = optixGetRayTmax();

	float3 hit_pos = ray_orgin + hit_t * ray_dir;

	optixSetPayload_0(prim_id);
	optixSetPayload_1(__float_as_uint(hit_pos.x));
	optixSetPayload_2(__float_as_uint(hit_pos.y));
	optixSetPayload_3(__float_as_uint(hit_pos.z));
}

//===================================================================
// approximates the contribution of the first vertex along the path 
// using SPH formulation 
//===================================================================
__forceinline__ __device__ float approxContribution(VPLSourcePoint& source, Particle& particle)
{
	OptixTraversableHandle handle = params.acc_structure;

	float3 pos = source.pos;

	uint32_t u[7] = {};
	u[0] = __float_as_uint(pos.x);
	u[1] = __float_as_uint(pos.y);
	u[2] = __float_as_uint(pos.z);
	u[6] = __float_as_uint(1e-16f);

	optixTrace(
		PAYLOAD_TYPE_BOUNDARY_VALUE,
		handle,
		pos, normalize(make_float3(1, 0, 0)),
		0.f, 1e-16f, 0.0f,
		OBJECT_FLUID,
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_BOUNDARY_VALUE,
		0, 0,
		u[0], u[1], u[2], u[3], u[4], u[5], u[6]);

	float3 boundary_value;
	boundary_value.x = __uint_as_float(u[3]);
	boundary_value.y = __uint_as_float(u[4]);
	boundary_value.z = __uint_as_float(u[5]);

	float norm_factor = __uint_as_float(u[6]);

	// ensure that the normal points outward from the fluid domain
	float3 source_normal = params.boundary.d_normal_buf[source.prim_id];
	if (dot(source_normal, source.pos - particle.pos) < 0.f) source_normal = -source_normal;
	particle.bNormal = source_normal;

	float contribution = source.area * 2.f * dot(source_normal, boundary_value / norm_factor);

	return contribution;
};

__forceinline__ __device__ BoundaryValuePRD loadBoundaryValuePRD()
{
	BoundaryValuePRD prd;
	prd.xi.x = __uint_as_float(optixGetPayload_0());
	prd.xi.y = __uint_as_float(optixGetPayload_1());
	prd.xi.z = __uint_as_float(optixGetPayload_2());

	prd.value.x = __uint_as_float(optixGetPayload_3());
	prd.value.y = __uint_as_float(optixGetPayload_4());
	prd.value.z = __uint_as_float(optixGetPayload_5());

	prd.norm_factor = __uint_as_float(optixGetPayload_6());

	return prd;
}

__forceinline__ __device__ void storeBoundaryValuePRD(const BoundaryValuePRD& prd)
{
	optixSetPayload_3(__float_as_uint(prd.value.x));
	optixSetPayload_4(__float_as_uint(prd.value.y));
	optixSetPayload_5(__float_as_uint(prd.value.z));

	optixSetPayload_6(__float_as_uint(prd.norm_factor));
}

extern "C" __global__ void __intersection__approx_boundary_value()
{
	optixSetPayloadTypes(PAYLOAD_TYPE_BOUNDARY_VALUE);

	uint32_t prim_idx = optixGetPrimitiveIndex();

	const float3 center = params.fluids.d_particles[prim_idx].pos;
	const float3 ray_origin = optixGetWorldRayOrigin();

	float3 r = ray_origin - center;
	float r2 = length2(r);
	float h = params.wcoeff.h;
	float h2 = h * h;

	if (r2 <= h2)
	{
		BoundaryValuePRD prd = loadBoundaryValuePRD();

		const Particle& pj = params.fluids.d_particles[prim_idx];
		const float3& xi = prd.xi;

		float3 xj = pj.pos;
		float3 vj = pj.vel;
		float3 volume_term_j = pj.volume_term;
		float Wij = W(xi - xj, params.wcoeff);
		float mj = params.fluids.mass0;
		float rho_j = pj.rho;

		prd.value += mj / rho_j * (volume_term_j + vj) * Wij;
		prd.norm_factor += Wij * mj / rho_j;
		storeBoundaryValuePRD(prd);
	}
}

extern "C" __global__ void __anyhit__approx_boundary_value()
{
	optixSetPayloadTypes(PAYLOAD_TYPE_BOUNDARY_VALUE);
	optixTerminateRay();
}

//=====================================
// neighbor search using ray tracing
//=====================================
extern "C" __global__ void __raygen__neighbor_search()
{
	const uint3 idx_3d = optixGetLaunchIndex();
	const uint3 dim_3d = optixGetLaunchDimensions();
	const uint32_t idx = flatten(idx_3d, dim_3d);

	const DeviceFluid& fluid = params.fluids;
	Particle& particle = params.fluids.d_particles[idx];

	if (idx >= fluid.num_particle) return;

	OptixTraversableHandle handle = params.acc_structure;

	const float3& point = params.fluids.d_particles[idx].pos;

	uint32_t u[2] = {};
	u[0] = idx;
	optixTrace(
		PAYLOAD_TYPE_NEIGHBOR_SEARCH,
		handle,
		point,
		normalize(make_float3(1, 0, 0)),
		0.f, 1e-16f, 0.0f,
		OBJECT_FLUID,
		OPTIX_RAY_FLAG_NONE,
		0,
		0,
		0,
		u[0], u[1]);

	particle.num_neighbor = u[1];
}

extern "C" __global__ void __intersection__neighbor_search()
{
	optixSetPayloadTypes(PAYLOAD_TYPE_NEIGHBOR_SEARCH);

	uint32_t prim_idx = optixGetPrimitiveIndex();

	const float3 center = params.fluids.d_particles[prim_idx].pos; //(*data).fluid.d_particles[prim_idx].pos;
	const float3 ray_origin = optixGetWorldRayOrigin();

	float3 r = ray_origin - center;
	float r2 = length2(r);
	float h = params.wcoeff.h; 
	float h2 = h * h;

	if (r2 <= h2)
	{
		const uint32_t idx = optixGetPayload_0();
		const uint32_t count = optixGetPayload_1();

		const uint32_t limit = params.config.num_neighbor;

		if (count < limit)
		{
			params.fluids.d_neighbor_id[idx * limit + count] = prim_idx;
		}

		if (count + 1 == limit)
			optixReportIntersection(0, 0);
		else
			optixSetPayload_1(count + 1);
	}
}

extern "C" __global__ void __anyhit__neighbor_search()
{
	optixSetPayloadTypes(PAYLOAD_TYPE_NEIGHBOR_SEARCH);
	optixTerminateRay();
}


//==========================
// pressure projection
//==========================

// Implementation of the walk-on-boundary method introduced by Sugimoto et al.
// https://doi.org/10.1145/3592109

__forceinline__ __device__
void projectVplConstruct(uint32_t idx, RandState& rand_state)
{
	const DeviceConfig& config = params.config;
	const DeviceBoundary& boundary = params.boundary;
	VPLRecord* vpl_data  = params.vpl_data + params.config.path_length * idx;
	Particle& particle   = params.fluids.d_particles[idx];
	uint32_t path_length = params.config.path_length;

	if(boundary.num_primitive <= 0) return;

	VPLSourcePoint source = sampleVPLSourcePoint(idx);

	if (source.prim_id == -1) { particle.bNormal = make_float3(0.f); return; }

	float contribution_n = approxContribution(source, particle);

	vpl_data[0] = { source.pos, contribution_n };

	VPLPoint y = { source.prim_id, source.pos };

	for (int i = 1; i < path_length; i++)
	{
		VPLPoint x = sampleVPLPoint(y, rand_state);
		if(x.prim_id == -1) continue;

		float3 n_x = boundary.d_normal_buf[x.prim_id];
		float sgn_x = dot(n_x, x.pos - y.pos) > 0.f ? -1.f : 1.f;

		contribution_n *= sgn_x;

		vpl_data[i] = { x.pos, contribution_n };

		y = { x.prim_id, x.pos };
	}

	vpl_data[config.path_length - 1].value *= 0.5f;
}

extern "C" __global__ void __raygen__project_vpl_construct()
{
	const uint3 idx_3d = optixGetLaunchIndex();
	const uint3 dim_3d = optixGetLaunchDimensions();
	const uint32_t idx = flatten(idx_3d, dim_3d);

	if (idx >= params.config.num_path_sample) return;
	RandState& rand_state = params.vpl_rand_states[idx];
	projectVplConstruct(idx, rand_state); 
}