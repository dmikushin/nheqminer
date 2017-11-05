namespace digit_4 {

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__global__ void kernel(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][(SSM - 1)];
	__shared__ uint ht_len[NRESTS];
	__shared__ uint4 lastword[NSLOTS];

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < NRESTS)
		ht_len[threadid] = 0;

	uint32_t bsize = umin(eq->edata.nslots[3][bucketid], NSLOTS);

	uint32_t hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	uint32_t si[2];
	uint4 tt[2];

	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	__syncthreads();

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
		pos[i] = atomicAdd(&ht_len[hr[i]], 1U);
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
				uint16_t prev = ht[hr[i]][k];
				int pair = __byte_perm(si[i], prev, 0x1054);
				uint32_t i = __byte_perm(pair, 0, 0x4510);
				uint32_t k = __byte_perm(pair, 0, 0x4532);

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
	}
}

} // namespace digit_4

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__forceinline__ void Digit_4(Equi<RB, SM>* equi)
{
	using namespace digit_4;

	kernel<RB, SM, SSM, PACKER, THREADS> << <NBUCKETS, THREADS >> >(equi);
}

