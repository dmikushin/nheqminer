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

#include <climits>
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
#include "digit_1.cuh"
#include "digit_2.cuh"
#include "digit_3.cuh"
#include "digit_4.cuh"
#include "digit_5.cuh"
#include "digit_6.cuh"
#include "digit_7.cuh"
#include "digit_8.cuh"
#include "digit_last_wdc.cuh"

template<uint32_t RB, uint32_t SM, uint32_t SSM, uint32_t THREADS, class PACKER>
class DjEzo : public ISolver
{
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
		CUDA_ERR_CHECK(cudaSetDevice(deviceID));
		CUDA_ERR_CHECK(cudaMemset(&equi->edata, 0, sizeof(equi->edata)));

		DigitFirst<RB, SM>(equi, tequihash_header, tequihash_header_len, nonce, nonce_len);

		Digit_1<RB, SM, SSM, 512>(equi);

		Digit_2<RB, SM, SSM, PACKER, THREADS>(equi);

		Digit_3<RB, SM, SSM, PACKER, THREADS>(equi);

		if (cancelf()) return;

		Digit_4<RB, SM, SSM, PACKER, THREADS>(equi);

		Digit_5<RB, SM, SSM, PACKER, THREADS>(equi);

		Digit_6<RB, SM, SSM - 1, PACKER>(equi);

		Digit_7<RB, SM, SSM - 1, PACKER>(equi);

		Digit_8<RB, SM, SSM - 1, PACKER>(equi);

		DigitLastWDC<RB, SM, SSM - 3, 2, PACKER, 64, 8, 4>(equi);

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
	static std::vector<std::unique_ptr<djezo::DjEzo<9, 1280, 12, 640, djezo::CantorPacker> > > solvers;
	if (!solvers.size())
	{
		int count = 0;
		CUDA_ERR_CHECK(cudaGetDeviceCount(&count));
		solvers.resize(count);
	}
	if (!solvers[deviceID].get())
	{
		solvers[deviceID].reset(new djezo::DjEzo<9, 1280, 12, 640, djezo::CantorPacker>(platformID, deviceID));
	}
	
	return dynamic_cast<ISolver*>(solvers[deviceID].get());
}

