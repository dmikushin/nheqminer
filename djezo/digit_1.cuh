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

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < 256)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;

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
	for (uint32_t s = threadIdx.x, plen = umin(pairs_len, MAXPAIRS); s < plen; s += blockDim.x)
	{
		int pair = pairs[s];
		uint32_t i = __byte_perm(pair, 0, 0x4510);
		uint32_t k = __byte_perm(pair, 0, 0x4532);

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

