namespace digit_1 {

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
template <uint32_t RB, uint32_t SM, int SSM, uint32_t THREADS>
__global__ void kernelDigit_1(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[256][SSM - 1];
	__shared__ uint ht_len[256 / 4]; // atomic adds on 1-byte lengths
	__shared__ uint2 lastword1[RB8_NSLOTS];
	__shared__ uint4 lastword2[RB8_NSLOTS];

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < 256 / 4)
		ht_len[threadid] = 0;

	uint32_t hr[2];
	uint8_t pos[2];
	pos[0] = pos[1] = SSM;

	uint2 ta[2];
	uint4 tb[2];

	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	__syncthreads();

	uint32_t bsize = umin(eq->edata.nslots0[bucketid], RB8_NSLOTS);

	#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		uint32_t si = i * THREADS + threadid;
		if (si >= bsize) break;

		const Slot* pslot1 = &eq->round0trees[bucketid][si];

		// get xhash
		tb[i] = *(uint4*)(&pslot1->hash[0]);
		ta[i] = *(uint2*)(&pslot1->hash[4]);
		lastword1[si] = ta[i];
		lastword2[si] = tb[i];

		asm("bfe.u32 %0, %1, 20, 8;" : "=r"(hr[i]) : "r"(ta[i].x));
		int shift = (hr[i] % 4) * 8;
		pos[i] = (atomicAdd(&ht_len[hr[i] / 4], 1U << shift) >> shift) & 0xff; // atomic adds on 1-byte lengths
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si;
	}

	__syncthreads();

	#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		if (pos[i] >= SSM) continue;

		for (int si = i * THREADS + threadid, k = 0, ke = pos[i], prev = ht[hr[i]][k], pair = 0, ii = si, kk = prev; k < ke;
			prev = ht[hr[i]][k],
			pair = __byte_perm(si, prev, 0x1054),
			ii = __byte_perm(pair, 0, 0x4510),
			kk = __byte_perm(pair, 0, 0x4532))
		{
			uint32_t xors[6];

			*(uint2*)(&xors[0]) = lastword1[ii] ^ lastword1[kk];

			uint32_t xorbucketid;
			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(RB), "r"(BUCKBITS));
			int xorslot = atomicAdd(&eq->edata.nslots[1][xorbucketid], 1);

			if (xorslot < NSLOTS)
			{
				*(uint4*)(&xors[2]) = lastword2[ii] ^ lastword2[kk];

				Slot &xs = eq->trees[0][xorbucketid][xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
				uint3 ttx;
				ttx.x = xors[5];
				ttx.y = xors[0];
				ttx.z = DefaultPacker::set_bucketid_and_slots(bucketid, ii, kk, 8, RB8_NSLOTS);
				*(uint3*)(&xs.hash[4]) = ttx;
			}
			
			k++;
			
			if (k == ke) break;
		}
	}
}

} // namespace digit_1

template <uint32_t RB, uint32_t SM, int SSM, uint32_t THREADS>
__forceinline__ void Digit_1(Equi<RB, SM>* equi)
{
	using namespace digit_1;

	kernelDigit_1<RB, SM, SSM, 512> << <4096, 512 >> >(equi);
}

