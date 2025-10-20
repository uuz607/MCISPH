#pragma once
#include <cuda_runtime.h>

namespace mcisph
{
	//https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
	// "Insert" a 0 bit after each of the 16 low bits of x
	__host__ __device__ inline uint32_t Part1By1(uint32_t x)
	{
		x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
		x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
		x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
		x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
		x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
		return x;
	}

	// "Insert" two 0 bits after each of the 10 low bits of x
	__host__ __device__ inline uint32_t Part1By2(uint32_t x)
	{
		x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
		x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
		x = (x ^ (x << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
		x = (x ^ (x << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
		x = (x ^ (x << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
		return x;
	}

	__host__ __device__ inline uint32_t MortonCode3(uint32_t x, uint32_t y, uint32_t z)
	{
		return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
	}

	// Inverse of Part1By1 - "delete" all odd-indexed bits
	__host__ __device__ inline uint32_t Compact1By1(uint32_t x)
	{
		x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
		x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
		x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
		x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
		x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
		return x;
	}

	// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
	__host__ __device__ inline uint32_t Compact1By2(uint32_t x)
	{
		x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
		x = (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
		x = (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
		x = (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
		x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
		return x;
	}

	__host__ __device__ inline uint32_t DecodeMorton2X(uint32_t code)
	{
		return Compact1By1(code >> 0);
	}

	__host__ __device__ inline uint32_t DecodeMorton2Y(uint32_t code)
	{
		return Compact1By1(code >> 1);
	}

	__host__ __device__ inline uint32_t DecodeMorton3X(uint32_t code)
	{
		return Compact1By2(code >> 0);
	}

	__host__ __device__ inline uint32_t DecodeMorton3Y(uint32_t code)
	{
		return Compact1By2(code >> 1);
	}

	__host__ __device__ inline uint32_t DecodeMorton3Z(uint32_t code)
	{
		return Compact1By2(code >> 2);
	}

	__host__ __device__ inline uint3 MortonCodeToIndex3(uint32_t mortonCode)
	{
		uint3 xyz;
		xyz.x = DecodeMorton3X(mortonCode);
		xyz.y = DecodeMorton3Y(mortonCode);
		xyz.z = DecodeMorton3Z(mortonCode);
		return xyz;
	}
	__host__ __device__ inline int3 MortonCodeToIndexint3(uint32_t mortonCode)
	{
		int3 xyz;
		xyz.x = (int)DecodeMorton3X(mortonCode);
		xyz.y = (int)DecodeMorton3Y(mortonCode);
		xyz.z = (int)DecodeMorton3Z(mortonCode);
		return xyz;
	}
}
