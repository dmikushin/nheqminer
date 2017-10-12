/*
  Equihash solver created by djeZo (l33tsoftw@gmail.com) for NiceHash

  Based on CUDA solver by John Tromp released under MIT license.

  Some helper functions taken out of OpenCL solver by Marc Bevand
  released under MIT license.

  cuda_djezo solver is released by NiceHash (www.nicehash.com) under
  GPL 3.0 license. If you don't have a copy, you can obtain one from
  https://www.gnu.org/licenses/gpl-3.0.txt
*/

/*
The MIT License (MIT)

Copyright (c) 2016 John Tromp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software, and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
The MIT License (MIT)

Copyright (c) 2016 Marc Bevand

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software, and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "check.h"
#include "blake2/blake2.h"
#include "djezo.hpp"

#include <memory>
#include <vector>

#define MAXREALSOLS 9
#define WN	200
#define WK	9
#define NDIGITS		(WK+1)
#define DIGITBITS	(WN/(NDIGITS))
#define PROOFSIZE (1<<WK)
#define BASE (1<<DIGITBITS)
#define NHASHES (2*BASE)
#define HASHESPERBLAKE (512/WN)
#define HASHOUT (HASHESPERBLAKE*WN/8)
#define NBLOCKS ((NHASHES + HASHESPERBLAKE - 1) / HASHESPERBLAKE)
#define BUCKBITS (DIGITBITS - RB)
#define NBUCKETS (1 << BUCKBITS)
#define BUCKMASK (NBUCKETS - 1)
#define SLOTBITS (RB + 2)
#define SLOTRANGE (1 << SLOTBITS)
#define NSLOTS SM
#define SLOTMASK (SLOTRANGE - 1)
#define NRESTS (1 << RB)
#define RESTMASK (NRESTS - 1)
#define CANTORBITS (2 * SLOTBITS - 2)
#define CANTORMASK ((1 << CANTORBITS) - 1)
#define CANTORMAXSQRT (2 * NSLOTS)
#define RB8_NSLOTS 640
#define RB8_NSLOTS_LD 624
#define FD_THREADS 128

namespace djezo {

struct SolutionsContainer
{
	uint32_t sols[MAXREALSOLS][512];
	uint32_t nsols;
};

struct __align__(32) Slot
{
	uint32_t hash[8];
};


struct __align__(16) SlotSmall
{
	uint32_t hash[4];
};


struct __align__(8) SlotTiny
{
	uint32_t hash[2];
};

template<uint32_t RB, uint32_t SM>
struct Equi
{
	Slot round0trees[4096][RB8_NSLOTS];
	Slot trees[1][NBUCKETS][NSLOTS];

	struct
	{
		SlotSmall treessmall[NSLOTS];
		SlotTiny treestiny[NSLOTS];
	}
	round2trees[NBUCKETS];

	struct
	{
		SlotSmall treessmall[NSLOTS];
		SlotTiny treestiny[NSLOTS];
	}
	round3trees[NBUCKETS];

	SlotSmall treessmall[4][NBUCKETS][NSLOTS];
	SlotTiny treestiny[1][4096][RB8_NSLOTS_LD];
	uint32_t round4bidandsids[NBUCKETS][NSLOTS];

	struct
	{
		uint32_t nslots8[4096];
		uint32_t nslots0[4096];
		uint32_t nslots[9][NBUCKETS];
		SolutionsContainer solutions;
	}
	edata;
};

struct DefaultPacker
{
	__device__ __forceinline__
	static uint32_t set_bucketid_and_slots(const uint32_t bucketid, const uint32_t s0, const uint32_t s1, const uint32_t RB, const uint32_t SM)
	{
		return (((bucketid << SLOTBITS) | s0) << SLOTBITS) | s1;
	}

	__device__ __forceinline__
	static uint32_t get_bucketid(const uint32_t bid, const uint32_t RB, const uint32_t SM)
	{
		// BUCKMASK-ed to prevent illegal memory accesses in case of memory errors
		return (bid >> (2 * SLOTBITS)) & BUCKMASK;
	}

	__device__ __forceinline__
	static uint32_t get_slot0(const uint32_t bid, const uint32_t s1, const uint32_t RB, const uint32_t SM)
	{
		return bid & SLOTMASK;
	}

	__device__ __forceinline__
	static uint32_t get_slot1(const uint32_t bid, const uint32_t RB, const uint32_t SM)
	{
		return (bid >> SLOTBITS) & SLOTMASK;
	}
};

struct CantorPacker
{
	__device__ __forceinline__
	static uint32_t cantor(const uint32_t s0, const uint32_t s1)
	{
		uint32_t a = umax(s0, s1);
		uint32_t b = umin(s0, s1);
		return a * (a + 1) / 2 + b;
	}

	__device__ __forceinline__
	static uint32_t set_bucketid_and_slots(const uint32_t bucketid, const uint32_t s0, const uint32_t s1, const uint32_t RB, const uint32_t SM)
	{
		return (bucketid << CANTORBITS) | cantor(s0, s1);
	}

	__device__ __forceinline__
	static uint32_t get_bucketid(const uint32_t bid, const uint32_t RB, const uint32_t SM)
	{
		return (bid >> CANTORBITS) & BUCKMASK;
	}

	__device__ __forceinline__
	static uint32_t get_slot0(const uint32_t bid, const uint32_t s1, const uint32_t RB, const uint32_t SM)
	{
		return ((bid & CANTORMASK) - cantor(0, s1)) & SLOTMASK;
	}

	__device__ __forceinline__
	static uint32_t get_slot1(const uint32_t bid, const uint32_t RB, const uint32_t SM)
	{
		uint32_t k, q, sqr = 8 * (bid & CANTORMASK) + 1;
		// this k=sqrt(sqr) computing loop averages 3.4 iterations out of maximum 9
		for (k = CANTORMAXSQRT; (q = sqr / k) < k; k = (k + q) / 2);
		return ((k - 1) / 2) & SLOTMASK;
	}
};

__device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b)
{
	return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

__device__ __forceinline__ uint4 operator^ (uint4 a, uint4 b)
{
	return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

#include "digit_first.cuh"

/*
  Functions digit_1 to digit_8 works by the same principle;
  Each thread does 2-3 slot loads (loads are coalesced). 
  Xorwork of slots is loaded into shared memory and is kept in registers (except for digit_1).
  At the same time, restbits (8 or 9 bits) in xorwork are used for collisions. 
  Restbits determine position in ht.
  Following next is pair creation. First one (or two) pairs' xorworks are put into global memory
  as soon as possible, the rest pairs are saved in shared memory (one uint32_t per pair - 16 bit indices). 
  In most cases, all threads have one (or two) pairs so with this trick, we offload memory writes a bit in last step.
  In last step we save xorwork of pairs in memory.
*/
template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS, uint32_t THREADS>
__global__ void digit_1(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[256][SSM - 1];
	__shared__ uint2 lastword1[RB8_NSLOTS];
	__shared__ uint4 lastword2[RB8_NSLOTS];
	__shared__ int ht_len[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < 256)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	uint32_t bsize = umin(eq->edata.nslots0[bucketid], RB8_NSLOTS);

	uint32_t hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	uint2 ta[2];
	uint4 tb[2];

	uint32_t si[2];

	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	//__syncthreads();

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		const Slot* pslot1 = eq->round0trees[bucketid] + si[i];

		// get xhash
		uint4 a1 = *(uint4*)(&pslot1->hash[0]);
		uint2 a2 = *(uint2*)(&pslot1->hash[4]);
		ta[i].x = a1.x;
		ta[i].y = a1.y;
		lastword1[si[i]] = ta[i];
		tb[i].x = a1.z;
		tb[i].y = a1.w;
		tb[i].z = a2.x;
		tb[i].w = a2.y;
		lastword2[si[i]] = tb[i];

		asm("bfe.u32 %0, %1, 20, 8;" : "=r"(hr[i]) : "r"(ta[i].x));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();
	int* pairs = ht_len;

	uint32_t xors[6];
	uint32_t xorbucketid, xorslot;

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			*(uint2*)(&xors[0]) = ta[i] ^ lastword1[p];

			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(RB), "r"(BUCKBITS));
			xorslot = atomicAdd(&eq->edata.nslots[1][xorbucketid], 1);

			if (xorslot < NSLOTS)
			{
				*(uint4*)(&xors[2]) = lastword2[si[i]] ^ lastword2[p];

				Slot &xs = eq->trees[0][xorbucketid][xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
				uint4 ttx;
				ttx.x = xors[5];
				ttx.y = xors[0];
				ttx.z = DefaultPacker::set_bucketid_and_slots(bucketid, si[i], p, 8, RB8_NSLOTS);
				ttx.w = 0;
				*(uint4*)(&xs.hash[4]) = ttx;
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				uint32_t pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				uint16_t prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	uint32_t plen = umin(pairs_len, MAXPAIRS);

	uint32_t i, k;
	for (uint32_t s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		*(uint2*)(&xors[0]) = lastword1[i] ^ lastword1[k];

		asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(RB), "r"(BUCKBITS));
		xorslot = atomicAdd(&eq->edata.nslots[1][xorbucketid], 1);

		if (xorslot < NSLOTS)
		{
			*(uint4*)(&xors[2]) = lastword2[i] ^ lastword2[k];

			Slot &xs = eq->trees[0][xorbucketid][xorslot];
			*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
			uint4 ttx;
			ttx.x = xors[5];
			ttx.y = xors[0];
			ttx.z = DefaultPacker::set_bucketid_and_slots(bucketid, i, k, 8, RB8_NSLOTS);
			ttx.w = 0;
			*(uint4*)(&xs.hash[4]) = ttx;
		}
	}
}

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS, uint32_t THREADS>
__global__ void digit_2(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][SSM - 1];
	__shared__ uint32_t lastword1[NSLOTS];
	__shared__ uint4 lastword2[NSLOTS];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < NRESTS)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	Slot* buck = eq->trees[0][bucketid];
	uint32_t bsize = umin(eq->edata.nslots[1][bucketid], NSLOTS);

	uint32_t hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	uint32_t ta[2];
	uint4 tt[2];

	uint32_t si[2];

	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	//__syncthreads();

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		// get slot
		const Slot* pslot1 = buck + si[i];

		uint4 ttx = *(uint4*)(&pslot1->hash[0]);
		lastword1[si[i]] = ta[i] = ttx.x;
		uint2 tty = *(uint2*)(&pslot1->hash[4]);
		tt[i].x = ttx.y;
		tt[i].y = ttx.z;
		tt[i].z = ttx.w;
		tt[i].w = tty.x;
		lastword2[si[i]] = tt[i];

		hr[i] = tty.y & RESTMASK;
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	uint32_t xors[5];
	uint32_t xorbucketid, xorslot;

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			xors[0] = ta[i] ^ lastword1[p];

			xorbucketid = xors[0] >> (12 + RB);
			xorslot = atomicAdd(&eq->edata.nslots[2][xorbucketid], 1);
			if (xorslot < NSLOTS)
			{
				*(uint4*)(&xors[1]) = tt[i] ^ lastword2[p];
				SlotSmall &xs = eq->round2trees[xorbucketid].treessmall[xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[0]);
				SlotTiny &xst = eq->round2trees[xorbucketid].treestiny[xorslot];
				uint2 ttx;
				ttx.x = xors[4];
				ttx.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
				*(uint2*)(&xst.hash[0]) = ttx;
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				uint32_t pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				uint16_t prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	uint32_t plen = umin(pairs_len, MAXPAIRS);

	uint32_t i, k;
	for (uint32_t s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		xors[0] = lastword1[i] ^ lastword1[k];

		xorbucketid = xors[0] >> (12 + RB);
		xorslot = atomicAdd(&eq->edata.nslots[2][xorbucketid], 1);
		if (xorslot < NSLOTS)
		{
			*(uint4*)(&xors[1]) = lastword2[i] ^ lastword2[k];
			SlotSmall &xs = eq->round2trees[xorbucketid].treessmall[xorslot];
			*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[0]);
			SlotTiny &xst = eq->round2trees[xorbucketid].treestiny[xorslot];
			uint2 ttx;
			ttx.x = xors[4];
			ttx.y = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
			*(uint2*)(&xst.hash[0]) = ttx;
		}
	}
}


template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS, uint32_t THREADS>
__global__ void digit_3(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][(SSM - 1)];
	__shared__ uint4 lastword1[NSLOTS];
	__shared__ uint32_t lastword2[NSLOTS];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < NRESTS)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	uint32_t bsize = umin(eq->edata.nslots[2][bucketid], NSLOTS);

	uint32_t hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	uint32_t si[2];
	uint4 tt[2];
	uint32_t ta[2];

	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	//__syncthreads();

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		SlotSmall &xs = eq->round2trees[bucketid].treessmall[si[i]];
		SlotTiny &xst = eq->round2trees[bucketid].treestiny[si[i]];

		tt[i] = *(uint4*)(&xs.hash[0]);
		lastword1[si[i]] = tt[i];
		ta[i] = xst.hash[0];
		lastword2[si[i]] = ta[i];
		asm("bfe.u32 %0, %1, 12, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	uint32_t xors[5];
	uint32_t bexor, xorbucketid, xorslot;

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			xors[4] = ta[i] ^ lastword2[p];

			if (xors[4] != 0)
			{
				*(uint4*)(&xors[0]) = tt[i] ^ lastword1[p];

				bexor = __byte_perm(xors[0], xors[1], 0x2107);
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(bexor), "r"(RB), "r"(BUCKBITS));
				xorslot = atomicAdd(&eq->edata.nslots[3][xorbucketid], 1);

				if (xorslot < NSLOTS)
				{
					SlotSmall &xs = eq->round3trees[xorbucketid].treessmall[xorslot];
					*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
					SlotTiny &xst = eq->round3trees[xorbucketid].treestiny[xorslot];
					uint2 ttx;
					ttx.x = bexor;
					ttx.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint2*)(&xst.hash[0]) = ttx;
				}
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				uint32_t pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				uint16_t prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	uint32_t plen = umin(pairs_len, MAXPAIRS);

	uint32_t i, k;
	for (uint32_t s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		xors[4] = lastword2[i] ^ lastword2[k];

		if (xors[4] != 0)
		{
			*(uint4*)(&xors[0]) = lastword1[i] ^ lastword1[k];

			bexor = __byte_perm(xors[0], xors[1], 0x2107);
			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(bexor), "r"(RB), "r"(BUCKBITS));
			xorslot = atomicAdd(&eq->edata.nslots[3][xorbucketid], 1);

			if (xorslot < NSLOTS)
			{
				SlotSmall &xs = eq->round3trees[xorbucketid].treessmall[xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
				SlotTiny &xst = eq->round3trees[xorbucketid].treestiny[xorslot];
				uint2 ttx;
				ttx.x = bexor;
				ttx.y = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
				*(uint2*)(&xst.hash[0]) = ttx;
			}
		}
	}
}


template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS, uint32_t THREADS>
__global__ void digit_4(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][(SSM - 1)];
	__shared__ uint4 lastword[NSLOTS];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < NRESTS)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	uint32_t bsize = umin(eq->edata.nslots[3][bucketid], NSLOTS);

	uint32_t hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	uint32_t si[2];
	uint4 tt[2];

	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	//__syncthreads();

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		SlotSmall &xs = eq->round3trees[bucketid].treessmall[si[i]];
		SlotTiny &xst = eq->round3trees[bucketid].treestiny[si[i]];

		// get xhash
		tt[i] = *(uint4*)(&xs.hash[0]);
		lastword[si[i]] = tt[i];
		hr[i] = xst.hash[0] & RESTMASK;
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();
	uint32_t xors[4];
	uint32_t xorbucketid, xorslot;

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			*(uint4*)(&xors[0]) = tt[i] ^ lastword[p];

			if (xors[3] != 0)
			{
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(4 + RB), "r"(BUCKBITS));
				xorslot = atomicAdd(&eq->edata.nslots[4][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					SlotSmall &xs = eq->treessmall[3][xorbucketid][xorslot];
					*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[0]);

					eq->round4bidandsids[xorbucketid][xorslot] = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
				}
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				uint32_t pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				uint16_t prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	uint32_t plen = umin(pairs_len, MAXPAIRS);
	uint32_t i, k;
	for (uint32_t s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		*(uint4*)(&xors[0]) = lastword[i] ^ lastword[k];
		if (xors[3] != 0)
		{
			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(4 + RB), "r"(BUCKBITS));
			xorslot = atomicAdd(&eq->edata.nslots[4][xorbucketid], 1);
			if (xorslot < NSLOTS)
			{
				SlotSmall &xs = eq->treessmall[3][xorbucketid][xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[0]);
				eq->round4bidandsids[xorbucketid][xorslot] = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
			}
		}
	}
}


template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS, uint32_t THREADS>
__global__ void digit_5(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][(SSM - 1)];
	__shared__ uint4 lastword[NSLOTS];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	if (threadid < NRESTS)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	SlotSmall* buck = eq->treessmall[3][bucketid];
	uint32_t bsize = umin(eq->edata.nslots[4][bucketid], NSLOTS);

	uint32_t hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	uint32_t si[2];
	uint4 tt[2];

	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	//__syncthreads();

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		const SlotSmall* pslot1 = buck + si[i];

		tt[i] = *(uint4*)(&pslot1->hash[0]);
		lastword[si[i]] = tt[i];
		asm("bfe.u32 %0, %1, 4, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();
	uint32_t xors[4];
	uint32_t bexor, xorbucketid, xorslot;

#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			*(uint4*)(&xors[0]) = tt[i] ^ lastword[p];

			if (xors[3] != 0)
			{
				bexor = __byte_perm(xors[0], xors[1], 0x1076);
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(bexor), "r"(RB), "r"(BUCKBITS));
				xorslot = atomicAdd(&eq->edata.nslots[5][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					SlotSmall &xs = eq->treessmall[2][xorbucketid][xorslot];
					uint4 ttx;
					ttx.x = xors[1];
					ttx.y = xors[2];
					ttx.z = xors[3];
					ttx.w = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint4*)(&xs.hash[0]) = ttx;
				}
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				uint32_t pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				uint16_t prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	uint32_t plen = umin(pairs_len, MAXPAIRS);
	uint32_t i, k;
	for (uint32_t s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		*(uint4*)(&xors[0]) = lastword[i] ^ lastword[k];

		if (xors[3] != 0)
		{
			bexor = __byte_perm(xors[0], xors[1], 0x1076);
			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(bexor), "r"(RB), "r"(BUCKBITS));
			xorslot = atomicAdd(&eq->edata.nslots[5][xorbucketid], 1);
			if (xorslot < NSLOTS)
			{
				SlotSmall &xs = eq->treessmall[2][xorbucketid][xorslot];
				uint4 tt;
				tt.x = xors[1];
				tt.y = xors[2];
				tt.z = xors[3];
				tt.w = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
				*(uint4*)(&xs.hash[0]) = tt;
			}
		}
	}
}


template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS>
__global__ void digit_6(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][(SSM - 1)];
	__shared__ uint2 lastword1[NSLOTS];
	__shared__ uint32_t lastword2[NSLOTS];
	__shared__ int ht_len[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t bsize_sh;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	ht_len[threadid] = 0;
	if (threadid == (NRESTS - 1))
	{
		pairs_len = 0;
		next_pair = 0;
	}
	else if (threadid == (NRESTS - 33))
		bsize_sh = umin(eq->edata.nslots[5][bucketid], NSLOTS);

	SlotSmall* buck = eq->treessmall[2][bucketid];

	uint32_t hr[3];
	int pos[3];
	pos[0] = pos[1] = pos[2] = SSM;

	uint32_t si[3];
	uint4 tt[3];

	__syncthreads();

	uint32_t bsize = bsize_sh;

#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		si[i] = i * NRESTS + threadid;
		if (si[i] >= bsize) break;

		const SlotSmall* pslot1 = buck + si[i];

		tt[i] = *(uint4*)(&pslot1->hash[0]);
		lastword1[si[i]] = *(uint2*)(&tt[i].x);
		lastword2[si[i]] = tt[i].z;
		asm("bfe.u32 %0, %1, 16, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	// doing this to save shared memory
	int* pairs = ht_len;
	__syncthreads();

	uint32_t xors[3];
	uint32_t bexor, xorbucketid, xorslot;

#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			xors[2] = tt[i].z ^ lastword2[p];

			if (xors[2] != 0)
			{
				*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ lastword1[p];

				bexor = __byte_perm(xors[0], xors[1], 0x1076);
				xorbucketid = bexor >> (12 + RB);
				xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					SlotSmall &xs = eq->treessmall[0][xorbucketid][xorslot];
					uint4 ttx;
					ttx.x = xors[1];
					ttx.y = xors[2];
					ttx.z = bexor;
					ttx.w = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint4*)(&xs.hash[0]) = ttx;
				}
			}

			if (pos[i] > 1)
			{
				p = ht[hr[i]][1];

				xors[2] = tt[i].z ^ lastword2[p];

				if (xors[2] != 0)
				{
					*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ lastword1[p];

					bexor = __byte_perm(xors[0], xors[1], 0x1076);
					xorbucketid = bexor >> (12 + RB);
					xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
					if (xorslot < NSLOTS)
					{
						SlotSmall &xs = eq->treessmall[0][xorbucketid][xorslot];
						uint4 ttx;
						ttx.x = xors[1];
						ttx.y = xors[2];
						ttx.z = bexor;
						ttx.w = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
						*(uint4*)(&xs.hash[0]) = ttx;
					}
				}

				for (int k = 2; k != pos[i]; ++k)
				{
					uint32_t pindex = atomicAdd(&pairs_len, 1);
					if (pindex >= MAXPAIRS) break;
					uint16_t prev = ht[hr[i]][k];
					pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
				}
			}
		}
	}

	__syncthreads();

	// process pairs
	uint32_t plen = umin(pairs_len, MAXPAIRS);
	for (uint32_t s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		uint32_t pair = pairs[s];
		uint32_t i = __byte_perm(pair, 0, 0x4510);
		uint32_t k = __byte_perm(pair, 0, 0x4532);

		xors[2] = lastword2[i] ^ lastword2[k];
		if (xors[2] == 0)
			continue;

		*(uint2*)(&xors[0]) = lastword1[i] ^ lastword1[k];

		bexor = __byte_perm(xors[0], xors[1], 0x1076);
		xorbucketid = bexor >> (12 + RB);
		xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
		if (xorslot >= NSLOTS) continue;
		SlotSmall &xs = eq->treessmall[0][xorbucketid][xorslot];
		uint4 ttx;
		ttx.x = xors[1];
		ttx.y = xors[2];
		ttx.z = bexor;
		ttx.w = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
		*(uint4*)(&xs.hash[0]) = ttx;
	}
}


template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS>
__global__ void digit_7(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][(SSM - 1)];
	__shared__ uint32_t lastword[NSLOTS][2];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t bsize_sh;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	ht_len[threadid] = 0;
	if (threadid == (NRESTS - 1))
	{
		pairs_len = 0;
		next_pair = 0;
	}
	else if (threadid == (NRESTS - 33))
		bsize_sh = umin(eq->edata.nslots[6][bucketid], NSLOTS);

	SlotSmall* buck = eq->treessmall[0][bucketid];

	uint32_t hr[3];
	int pos[3];
	pos[0] = pos[1] = pos[2] = SSM;

	uint32_t si[3];
	uint4 tt[3];

	__syncthreads();

	uint32_t bsize = bsize_sh;

#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		si[i] = i * NRESTS + threadid;
		if (si[i] >= bsize) break;

		const SlotSmall* pslot1 = buck + si[i];

		// get xhash
		tt[i] = *(uint4*)(&pslot1->hash[0]);
		*(uint2*)(&lastword[si[i]][0]) = *(uint2*)(&tt[i].x);
		asm("bfe.u32 %0, %1, 12, %2;" : "=r"(hr[i]) : "r"(tt[i].z), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	uint32_t xors[2];
	uint32_t xorbucketid, xorslot;

#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

			if (xors[1] != 0)
			{
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(8 + RB), "r"(BUCKBITS));
				xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					SlotSmall &xs = eq->treessmall[1][xorbucketid][xorslot];
					uint4 ttx;
					ttx.x = xors[0];
					ttx.y = xors[1];
					ttx.z = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					ttx.w = 0;
					*(uint4*)(&xs.hash[0]) = ttx;
				}
			}

			if (pos[i] > 1)
			{
				p = ht[hr[i]][1];

				*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

				if (xors[1] != 0)
				{
					asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(8 + RB), "r"(BUCKBITS));
					xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
					if (xorslot < NSLOTS)
					{
						SlotSmall &xs = eq->treessmall[1][xorbucketid][xorslot];
						uint4 ttx;
						ttx.x = xors[0];
						ttx.y = xors[1];
						ttx.z = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
						ttx.w = 0;
						*(uint4*)(&xs.hash[0]) = ttx;
					}
				}

				for (int k = 2; k != pos[i]; ++k)
				{
					uint32_t pindex = atomicAdd(&pairs_len, 1);
					if (pindex >= MAXPAIRS) break;
					uint16_t prev = ht[hr[i]][k];
					pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
				}
			}
		}
	}

	__syncthreads();

	// process pairs
	uint32_t plen = umin(pairs_len, MAXPAIRS);
	for (uint32_t s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		uint32_t i = __byte_perm(pair, 0, 0x4510);
		uint32_t k = __byte_perm(pair, 0, 0x4532);

		*(uint2*)(&xors[0]) = *(uint2*)(&lastword[i][0]) ^ *(uint2*)(&lastword[k][0]);

		if (xors[1] == 0)
			continue;

		asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(8 + RB), "r"(BUCKBITS));
		xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
		if (xorslot >= NSLOTS) continue;
		SlotSmall &xs = eq->treessmall[1][xorbucketid][xorslot];
		uint4 tt;
		tt.x = xors[0];
		tt.y = xors[1];
		tt.z = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
		tt.w = 0;
		*(uint4*)(&xs.hash[0]) = tt;
	}
}


template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS>
__global__ void digit_8(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][(SSM - 1)];
	__shared__ uint32_t lastword[NSLOTS][2];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t bsize_sh;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	ht_len[threadid] = 0;
	if (threadid == (NRESTS - 1))
	{
		next_pair = 0;
		pairs_len = 0;
	}
	else if (threadid == (NRESTS - 33))
		bsize_sh = umin(eq->edata.nslots[7][bucketid], NSLOTS);

	SlotSmall* buck = eq->treessmall[1][bucketid];

	uint32_t hr[3];
	int pos[3];
	pos[0] = pos[1] = pos[2] = SSM;

	uint32_t si[3];
	uint2 tt[3];

	__syncthreads();

	uint32_t bsize = bsize_sh;

#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		si[i] = i * NRESTS + threadid;
		if (si[i] >= bsize) break;

		const SlotSmall* pslot1 = buck + si[i];

		// get xhash
		tt[i] = *(uint2*)(&pslot1->hash[0]);
		*(uint2*)(&lastword[si[i]][0]) = *(uint2*)(&tt[i].x);
		asm("bfe.u32 %0, %1, 8, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	uint32_t xors[2];
	uint32_t bexor, xorbucketid, xorslot;

#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

			if (xors[1] != 0)
			{
				bexor = __byte_perm(xors[0], xors[1], 0x0765);
				xorbucketid = bexor >> (12 + 8);
				xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
				if (xorslot < RB8_NSLOTS_LD)
				{
					SlotTiny &xs = eq->treestiny[0][xorbucketid][xorslot];
					uint2 tt;
					tt.x = xors[1];
					tt.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint2*)(&xs.hash[0]) = tt;
				}
			}

			if (pos[i] > 1)
			{
				p = ht[hr[i]][1];

				*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

				if (xors[1] != 0)
				{
					bexor = __byte_perm(xors[0], xors[1], 0x0765);
					xorbucketid = bexor >> (12 + 8);
					xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
					if (xorslot < RB8_NSLOTS_LD)
					{
						SlotTiny &xs = eq->treestiny[0][xorbucketid][xorslot];
						uint2 tt;
						tt.x = xors[1];
						tt.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
						*(uint2*)(&xs.hash[0]) = tt;
					}
				}

				for (int k = 2; k != pos[i]; ++k)
				{
					uint32_t pindex = atomicAdd(&pairs_len, 1);
					if (pindex >= MAXPAIRS) break;
					uint16_t prev = ht[hr[i]][k];
					pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
				}
			}
		}
	}

	__syncthreads();

	// process pairs
	uint32_t plen = umin(pairs_len, MAXPAIRS);
	for (uint32_t s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		uint32_t i = __byte_perm(pair, 0, 0x4510);
		uint32_t k = __byte_perm(pair, 0, 0x4532);

		*(uint2*)(&xors[0]) = *(uint2*)(&lastword[i][0]) ^ *(uint2*)(&lastword[k][0]);

		if (xors[1] == 0)
			continue;

		bexor = __byte_perm(xors[0], xors[1], 0x0765);
		xorbucketid = bexor >> (12 + 8);
		xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
		if (xorslot >= RB8_NSLOTS_LD) continue;
		SlotTiny &xs = eq->treestiny[0][xorbucketid][xorslot];
		uint2 tt;
		tt.x = xors[1];
		tt.y = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
		*(uint2*)(&xs.hash[0]) = tt;
	}
}

/*
  Last round function is similar to previous ones but has different ending.
  We use warps to process final candidates. Each warp process one candidate.
  First two bidandsids (uint32_t of stored bucketid and two slotids) are retreived by
  lane 0 and lane 16, next four bidandsids by lane 0, 8, 16 and 24, ... until
  all lanes in warp have bidandsids from round 4. Next, each thread retreives
  16 indices. While doing so, indices are put into comparison using atomicExch
  to determine if there are duplicates (tromp's method). At the end, if no
  duplicates are found, candidate solution is saved (all indices). Note that this
  dup check method is not exact so CPU dup checking is needed after.
*/
template <uint32_t RB, uint32_t SM, int SSM, uint32_t FCT, typename PACKER, uint32_t MAXPAIRS, uint32_t DUPBITS, uint32_t W>
__global__ void digit_last_wdc(Equi<RB, SM>* eq)
{
	__shared__ uint8_t shared_data[8192];
	int* ht_len = (int*)(&shared_data[0]);
	int* pairs = ht_len;
	uint32_t* lastword = (uint32_t*)(&shared_data[256 * 4]);
	uint16_t* ht = (uint16_t*)(&shared_data[256 * 4 + RB8_NSLOTS_LD * 4]);
	uint32_t* pairs_len = (uint32_t*)(&shared_data[8188]);

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
#pragma unroll
	for (uint32_t i = 0; i != FCT; ++i)
		ht_len[(i * (256 / FCT)) + threadid] = 0;

	if (threadid == ((256 / FCT) - 1))
		*pairs_len = 0;

	SlotTiny* buck = eq->treestiny[0][bucketid];
	uint32_t bsize = umin(eq->edata.nslots8[bucketid], RB8_NSLOTS_LD);

	uint32_t si[3 * FCT];
	uint32_t hr[3 * FCT];
	int pos[3 * FCT];
	uint32_t lw[3 * FCT];
#pragma unroll
	for (uint32_t i = 0; i != (3 * FCT); ++i)
		pos[i] = SSM;

	__syncthreads();

#pragma unroll
	for (uint32_t i = 0; i != (3 * FCT); ++i)
	{
		si[i] = i * (256 / FCT) + threadid;
		if (si[i] >= bsize) break;

		const SlotTiny* pslot1 = buck + si[i];

		// get xhash
		uint2 tt = *(uint2*)(&pslot1->hash[0]);
		lw[i] = tt.x;
		lastword[si[i]] = lw[i];

		uint32_t a;
		asm("bfe.u32 %0, %1, 20, 8;" : "=r"(a) : "r"(lw[i]));
		hr[i] = a;

		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1))
			ht[hr[i] * (SSM - 1) + pos[i]] = si[i];
	}

	__syncthreads();

#pragma unroll
	for (uint32_t i = 0; i != (3 * FCT); ++i)
	{
		if (pos[i] >= SSM) continue;

		for (int k = 0; k != pos[i]; ++k)
		{
			uint16_t prev = ht[hr[i] * (SSM - 1) + k];
			if (lw[i] != lastword[prev]) continue;
			uint32_t pindex = atomicAdd(pairs_len, 1);
			if (pindex >= MAXPAIRS) break;
			pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
		}
	}

	__syncthreads();
	uint32_t plen = umin(*pairs_len, 64);

#define CALC_LEVEL(a, b, c, d) { \
	uint32_t plvl = levels[b]; \
	uint32_t* bucks = eq->round4bidandsids[PACKER::get_bucketid(plvl, RB, SM)]; \
	uint32_t slot1 = PACKER::get_slot1(plvl, RB, SM); \
	uint32_t slot0 = PACKER::get_slot0(plvl, slot1, RB, SM); \
	levels[b] = bucks[slot1]; \
	levels[c] = bucks[slot0]; \
				}

#define CALC_LEVEL_SMALL(a, b, c, d) { \
	uint32_t plvl = levels[b]; \
	SlotSmall* bucks = eq->treessmall[a][PACKER::get_bucketid(plvl, RB, SM)]; \
	uint32_t slot1 = PACKER::get_slot1(plvl, RB, SM); \
	uint32_t slot0 = PACKER::get_slot0(plvl, slot1, RB, SM); \
	levels[b] = bucks[slot1].hash[d]; \
	levels[c] = bucks[slot0].hash[d]; \
				}

	uint32_t lane = threadIdx.x & 0x1f;
	uint32_t par = threadIdx.x >> 5;

	uint32_t* levels = (uint32_t*)&pairs[MAXPAIRS + (par << DUPBITS)];
	uint32_t* susp = levels;

	while (par < plen)
	{
		int pair = pairs[par];
		par += W;

		if (lane % 16 == 0)
		{
			uint32_t plvl;
			if (lane == 0) plvl = buck[__byte_perm(pair, 0, 0x4510)].hash[1];
			else plvl = buck[__byte_perm(pair, 0, 0x4532)].hash[1];
			SlotSmall* bucks = eq->treessmall[1][PACKER::get_bucketid(plvl, RB, SM)];
			uint32_t slot1 = PACKER::get_slot1(plvl, RB, SM);
			uint32_t slot0 = PACKER::get_slot0(plvl, slot1, RB, SM);
			levels[lane] = bucks[slot1].hash[2];
			levels[lane + 8] = bucks[slot0].hash[2];
		}

		if (lane % 8 == 0)
			CALC_LEVEL_SMALL(0, lane, lane + 4, 3);

		if (lane % 4 == 0)
			CALC_LEVEL_SMALL(2, lane, lane + 2, 3);

		if (lane % 2 == 0)
			CALC_LEVEL(0, lane, lane + 1, 4);

		uint32_t ind[16];

		uint32_t f1 = levels[lane];
		const SlotTiny* buck_v4 = &eq->round3trees[PACKER::get_bucketid(f1, RB, SM)].treestiny[0];
		const uint32_t slot1_v4 = PACKER::get_slot1(f1, RB, SM);
		const uint32_t slot0_v4 = PACKER::get_slot0(f1, slot1_v4, RB, SM);

		susp[lane] = 0xffffffff;
		susp[32 + lane] = 0xffffffff;

#define CHECK_DUP(a) \
	__any(atomicExch(&susp[(ind[a] & ((1 << DUPBITS) - 1))], (ind[a] >> DUPBITS)) == (ind[a] >> DUPBITS))

		uint32_t f2 = buck_v4[slot1_v4].hash[1];
		const SlotTiny* buck_v3_1 = &eq->round2trees[PACKER::get_bucketid(f2, RB, SM)].treestiny[0];
		const uint32_t slot1_v3_1 = PACKER::get_slot1(f2, RB, SM);
		const uint32_t slot0_v3_1 = PACKER::get_slot0(f2, slot1_v3_1, RB, SM);

		susp[64 + lane] = 0xffffffff;
		susp[96 + lane] = 0xffffffff;

		uint32_t f0 = buck_v3_1[slot1_v3_1].hash[1];
		const Slot* buck_v2_1 = eq->trees[0][PACKER::get_bucketid(f0, RB, SM)];
		const uint32_t slot1_v2_1 = PACKER::get_slot1(f0, RB, SM);
		const uint32_t slot0_v2_1 = PACKER::get_slot0(f0, slot1_v2_1, RB, SM);

		susp[128 + lane] = 0xffffffff;
		susp[160 + lane] = 0xffffffff;

		uint32_t f3 = buck_v2_1[slot1_v2_1].hash[6];
		const Slot* buck_fin_1 = eq->round0trees[DefaultPacker::get_bucketid(f3, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_1 = DefaultPacker::get_slot1(f3, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_1 = DefaultPacker::get_slot0(f3, slot1_fin_1, 8, RB8_NSLOTS);

		susp[192 + lane] = 0xffffffff;
		susp[224 + lane] = 0xffffffff;

		ind[0] = buck_fin_1[slot1_fin_1].hash[7];
		if (CHECK_DUP(0)) continue;
		ind[1] = buck_fin_1[slot0_fin_1].hash[7];
		if (CHECK_DUP(1)) continue;

		uint32_t f4 = buck_v2_1[slot0_v2_1].hash[6];
		const Slot* buck_fin_2 = eq->round0trees[DefaultPacker::get_bucketid(f4, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_2 = DefaultPacker::get_slot1(f4, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_2 = DefaultPacker::get_slot0(f4, slot1_fin_2, 8, RB8_NSLOTS);

		ind[2] = buck_fin_2[slot1_fin_2].hash[7];
		if (CHECK_DUP(2)) continue;
		ind[3] = buck_fin_2[slot0_fin_2].hash[7];
		if (CHECK_DUP(3)) continue;

		uint32_t f5 = buck_v3_1[slot0_v3_1].hash[1];
		const Slot* buck_v2_2 = eq->trees[0][PACKER::get_bucketid(f5, RB, SM)];
		const uint32_t slot1_v2_2 = PACKER::get_slot1(f5, RB, SM);
		const uint32_t slot0_v2_2 = PACKER::get_slot0(f5, slot1_v2_2, RB, SM);

		uint32_t f6 = buck_v2_2[slot1_v2_2].hash[6];
		const Slot* buck_fin_3 = eq->round0trees[DefaultPacker::get_bucketid(f6, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_3 = DefaultPacker::get_slot1(f6, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_3 = DefaultPacker::get_slot0(f6, slot1_fin_3, 8, RB8_NSLOTS);

		ind[4] = buck_fin_3[slot1_fin_3].hash[7];
		if (CHECK_DUP(4)) continue;
		ind[5] = buck_fin_3[slot0_fin_3].hash[7];
		if (CHECK_DUP(5)) continue;

		uint32_t f7 = buck_v2_2[slot0_v2_2].hash[6];
		const Slot* buck_fin_4 = eq->round0trees[DefaultPacker::get_bucketid(f7, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_4 = DefaultPacker::get_slot1(f7, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_4 = DefaultPacker::get_slot0(f7, slot1_fin_4, 8, RB8_NSLOTS);

		ind[6] = buck_fin_4[slot1_fin_4].hash[7];
		if (CHECK_DUP(6)) continue;
		ind[7] = buck_fin_4[slot0_fin_4].hash[7];
		if (CHECK_DUP(7)) continue;

		uint32_t f8 = buck_v4[slot0_v4].hash[1];
		const SlotTiny* buck_v3_2 = &eq->round2trees[PACKER::get_bucketid(f8, RB, SM)].treestiny[0];
		const uint32_t slot1_v3_2 = PACKER::get_slot1(f8, RB, SM);
		const uint32_t slot0_v3_2 = PACKER::get_slot0(f8, slot1_v3_2, RB, SM);

		uint32_t f9 = buck_v3_2[slot1_v3_2].hash[1];
		const Slot* buck_v2_3 = eq->trees[0][PACKER::get_bucketid(f9, RB, SM)];
		const uint32_t slot1_v2_3 = PACKER::get_slot1(f9, RB, SM);
		const uint32_t slot0_v2_3 = PACKER::get_slot0(f9, slot1_v2_3, RB, SM);

		uint32_t f10 = buck_v2_3[slot1_v2_3].hash[6];
		const Slot* buck_fin_5 = eq->round0trees[DefaultPacker::get_bucketid(f10, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_5 = DefaultPacker::get_slot1(f10, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_5 = DefaultPacker::get_slot0(f10, slot1_fin_5, 8, RB8_NSLOTS);

		ind[8] = buck_fin_5[slot1_fin_5].hash[7];
		if (CHECK_DUP(8)) continue;
		ind[9] = buck_fin_5[slot0_fin_5].hash[7];
		if (CHECK_DUP(9)) continue;

		uint32_t f11 = buck_v2_3[slot0_v2_3].hash[6];
		const Slot* buck_fin_6 = eq->round0trees[DefaultPacker::get_bucketid(f11, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_6 = DefaultPacker::get_slot1(f11, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_6 = DefaultPacker::get_slot0(f11, slot1_fin_6, 8, RB8_NSLOTS);

		ind[10] = buck_fin_6[slot1_fin_6].hash[7];
		if (CHECK_DUP(10)) continue;
		ind[11] = buck_fin_6[slot0_fin_6].hash[7];
		if (CHECK_DUP(11)) continue;

		uint32_t f12 = buck_v3_2[slot0_v3_2].hash[1];
		const Slot* buck_v2_4 = eq->trees[0][PACKER::get_bucketid(f12, RB, SM)];
		const uint32_t slot1_v2_4 = PACKER::get_slot1(f12, RB, SM);
		const uint32_t slot0_v2_4 = PACKER::get_slot0(f12, slot1_v2_4, RB, SM);

		uint32_t f13 = buck_v2_4[slot1_v2_4].hash[6];
		const Slot* buck_fin_7 = eq->round0trees[DefaultPacker::get_bucketid(f13, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_7 = DefaultPacker::get_slot1(f13, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_7 = DefaultPacker::get_slot0(f13, slot1_fin_7, 8, RB8_NSLOTS);

		ind[12] = buck_fin_7[slot1_fin_7].hash[7];
		if (CHECK_DUP(12)) continue;
		ind[13] = buck_fin_7[slot0_fin_7].hash[7];
		if (CHECK_DUP(13)) continue;

		uint32_t f14 = buck_v2_4[slot0_v2_4].hash[6];
		const Slot* buck_fin_8 = eq->round0trees[DefaultPacker::get_bucketid(f14, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_8 = DefaultPacker::get_slot1(f14, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_8 = DefaultPacker::get_slot0(f14, slot1_fin_8, 8, RB8_NSLOTS);

		ind[14] = buck_fin_8[slot1_fin_8].hash[7];
		if (CHECK_DUP(14)) continue;
		ind[15] = buck_fin_8[slot0_fin_8].hash[7];
		if (CHECK_DUP(15)) continue;

		uint32_t soli;
		if (lane == 0)
		{
			soli = atomicAdd(&eq->edata.solutions.nsols, 1);
		}
		soli = __shfl(soli, 0);

		if (soli < MAXREALSOLS)
		{
			uint32_t pos = lane << 4;
			*(uint4*)(&eq->edata.solutions.sols[soli][pos]) = *(uint4*)(&ind[0]);
			*(uint4*)(&eq->edata.solutions.sols[soli][pos + 4]) = *(uint4*)(&ind[4]);
			*(uint4*)(&eq->edata.solutions.sols[soli][pos + 8]) = *(uint4*)(&ind[8]);
			*(uint4*)(&eq->edata.solutions.sols[soli][pos + 12]) = *(uint4*)(&ind[12]);
		}
	}
}

template<uint32_t RB, uint32_t SM, uint32_t SSM, uint32_t THREADS, class PACKER>
class DjEzo : public ISolver
{
	int threadsperblock;
	int blocks;
	
	std::vector<SolutionsContainer> vsolutions;
	SolutionsContainer* solutions;

	Equi<RB, SM>* equi;

public:

	const int platformID, deviceID;

	DjEzo(int platformID_, int deviceID_) :
		platformID(platformID_), deviceID(deviceID_),
		equi(NULL)
	{
		cudaDeviceProp props;
		CUDA_ERR_CHECK(cudaGetDeviceProperties(&props, deviceID));

		if (props.major < 5)
			throw std::runtime_error("Only CUDA devices with SM 5.0 and higher are supported.");

		CUDA_ERR_CHECK(cudaSetDevice(deviceID));
		CUDA_ERR_CHECK(cudaDeviceReset());
		CUDA_ERR_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

		CUDA_ERR_CHECK(cudaMalloc((void**)&equi, sizeof(Equi<RB, SM>)));

		vsolutions.resize(1);
		solutions = &vsolutions[0];
	}

	virtual ~DjEzo()
	{
		CUDA_ERR_CHECK(cudaFree(equi));
	}

	virtual void start() { }
	virtual void stop() { }

	static void setheader(blake2b_state *ctx, const char *header, const uint32_t headerLen, const char* nce, const uint32_t nonceLen)
	{
		uint32_t le_N = WN;
		uint32_t le_K = WK;
		uint8_t personal[] = "ZcashPoW01230123";
		memcpy(personal + 8, &le_N, 4);
		memcpy(personal + 12, &le_K, 4);
		blake2b_param P[1];
		P->digest_length = HASHOUT;
		P->key_length = 0;
		P->fanout = 1;
		P->depth = 1;
		P->leaf_length = 0;
		P->node_offset = 0;
		P->node_depth = 0;
		P->inner_length = 0;
		memset(P->reserved, 0, sizeof(P->reserved));
		memset(P->salt, 0, sizeof(P->salt));
		memcpy(P->personal, (const uint8_t *)personal, 16);
		blake2b_init_param(ctx, P);
		blake2b_update(ctx, (const uint8_t *)header, headerLen);
		blake2b_update(ctx, (const uint8_t *)nce, nonceLen);
	}

	static int compu32(const void *pa, const void *pb)
	{
		uint32_t a = *(uint32_t *)pa, b = *(uint32_t *)pb;
		return a<b ? -1 : a == b ? 0 : +1;
	}

	static bool duped(uint32_t* prf)
	{
		uint32_t sortprf[512];
		memcpy(sortprf, prf, sizeof(uint32_t) * 512);
		qsort(sortprf, 512, sizeof(uint32_t), &compu32);
		for (uint32_t i = 1; i<512; i++)
			if (sortprf[i] <= sortprf[i - 1])
				return true;
		return false;
	}

	static void sort_pair(uint32_t *a, uint32_t len)
	{
		uint32_t *b = a + len;
		uint32_t tmp, need_sorting = 0;
		for (uint32_t i = 0; i < len; i++)
			if (need_sorting || a[i] > b[i])
			{
				need_sorting = 1;
				tmp = a[i];
				a[i] = b[i];
				b[i] = tmp;
			}
			else if (a[i] < b[i])
				return;
	}

	virtual void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		std::function<bool()> cancelf,
		std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef)
	{
		int blocks = NBUCKETS;

		CUDA_ERR_CHECK(cudaMemset(&equi->edata, 0, sizeof(equi->edata)));

		DigitFirst<RB, SM>(equi, tequihash_header, tequihash_header_len, nonce, nonce_len);

		digit_1<RB, SM, SSM, PACKER, 4 * NRESTS, 512> << <4096, 512 >> >(equi);

		digit_2<RB, SM, SSM, PACKER, 4 * NRESTS, THREADS> << <blocks, THREADS >> >(equi);

		digit_3<RB, SM, SSM, PACKER, 4 * NRESTS, THREADS> << <blocks, THREADS >> >(equi);

		if (cancelf()) return;

		digit_4<RB, SM, SSM, PACKER, 4 * NRESTS, THREADS> << <blocks, THREADS >> >(equi);

		digit_5<RB, SM, SSM, PACKER, 4 * NRESTS, THREADS> << <blocks, THREADS >> >(equi);

		digit_6<RB, SM, SSM - 1, PACKER, 3 * NRESTS> << <blocks, NRESTS >> >(equi);

		digit_7<RB, SM, SSM - 1, PACKER, 3 * NRESTS> << <blocks, NRESTS >> >(equi);

		digit_8<RB, SM, SSM - 1, PACKER, 3 * NRESTS> << <blocks, NRESTS >> >(equi);

		digit_last_wdc<RB, SM, SSM - 3, 2, PACKER, 64, 8, 4> << <4096, 256 / 2 >> >(equi);

		CUDA_ERR_CHECK(cudaMemcpy(solutions, &equi->edata.solutions, (MAXREALSOLS * (512 * 4)) + 4, cudaMemcpyDeviceToHost));

		//printf("nsols: %u\n", solutions->nsols);
		//if (solutions->nsols > 9)
		//	printf("missing sol, total: %u\n", solutions->nsols);

		for (uint32_t s = 0; (s < solutions->nsols) && (s < MAXREALSOLS); s++)
		{
			// remove dups on CPU (dup removal on GPU is not fully exact and can pass on some invalid solutions)
			if (duped(solutions->sols[s])) continue;

			// perform sort of pairs
			for (uint32_t level = 0; level < 9; level++)
				for (uint32_t i = 0; i < (1 << 9); i += (2 << level))
					sort_pair(&solutions->sols[s][i], 1 << level);

			std::vector<uint32_t> index_vector(PROOFSIZE);
			for (uint32_t i = 0; i < PROOFSIZE; i++) {
				index_vector[i] = solutions->sols[s][i];
			}
		
			solutionf(index_vector, DIGITBITS, nullptr);
		}

		hashdonef();
	}

	virtual std::string getdevinfo()
	{
		cudaDeviceProp props;
		CUDA_ERR_CHECK(cudaGetDeviceProperties(&props, deviceID));

		return (std::string)props.name + " (#" + std::to_string(deviceID) + ")";
	}

	virtual std::string getname() { return "CUDA-DJEZO"; }

	virtual SolverType GetType() const { return SolverType::CUDA; }
};

} // namespace djezo

extern "C" ISolver* djezoSolver(int platformID, int deviceID)
{
	static std::unique_ptr<djezo::DjEzo<9, 1248, 12, 640, djezo::CantorPacker> > solver;
	if (solver.get())
	{
		if ((solver->platformID != platformID) || (solver->deviceID != deviceID))
			solver.reset(NULL);
	}
	if (!solver.get())
	{
		solver.reset(new djezo::DjEzo<9, 1248, 12, 640, djezo::CantorPacker>(platformID, deviceID));
	}
	
	return dynamic_cast<ISolver*>(solver.get());
}

