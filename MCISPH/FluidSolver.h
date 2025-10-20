#pragma once
#include "Geometry.cuh"
#include "SPHKernel.cuh"

namespace mcisph
{
	// payload type ID constants 
	constexpr OptixPayloadTypeID PAYLOAD_TYPE_SOURCE_POINT = OPTIX_PAYLOAD_TYPE_ID_0;
	constexpr OptixPayloadTypeID PAYLOAD_TYPE_IMMEDIATE_POINT = PAYLOAD_TYPE_SOURCE_POINT;
	constexpr OptixPayloadTypeID PAYLOAD_TYPE_BOUNDARY_VALUE = OPTIX_PAYLOAD_TYPE_ID_1;
	constexpr OptixPayloadTypeID PAYLOAD_TYPE_NEIGHBOR_SEARCH = OPTIX_PAYLOAD_TYPE_ID_2;

	struct VPLRecord
	{
		float3 pos;
		float value;
	};

	struct Params
	{
		OptixTraversableHandle acc_structure;
		DeviceFluid fluids;
		DeviceBoundary boundary;
		RandState* vpl_rand_states;
		VPLRecord* vpl_data;
		SPHKernelWCoeff wcoeff;
		DeviceConfig config;
	};

	// payload: sample vpl points
	struct VPLPoint
	{	
		uint32_t prim_id;
		float3 pos;
	};

	struct VPLSourcePoint
	{	
		uint32_t prim_id;
		float3 pos;
		float area;
	};

	const uint32_t vplPointPayloadSemantics[4] =
	{
		//VPLPoint::prim_id
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
		//VPLPoint::pos
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE
	};

	// payload: approximate the boundary value of the first vertex along the path
	struct BoundaryValuePRD
	{
		float3 xi;
		float3 value;
		float norm_factor;
	};

	const uint32_t boundaryValuePRDSemantics[7] =
	{
		// BoundaryValuePRD::xi
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ,
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ,
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ,
		//BoundaryValuePRD::value
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE,
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE,
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE,
		//BoundaryValuePRD::norm_factor
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE
	};

	// payload: neighbor Search 
	struct NeighborSearchPRD
	{
		uint32_t idx;
		uint32_t count;
	};

	const uint32_t neighborSearchPRDSemantics[2] =
	{
		// NeighborSearchPRD::xi
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE,
		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE
	};

}