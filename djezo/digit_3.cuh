namespace digit_3 {

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__global__ void kernel(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][SSM - 1];
	__shared__ uint ht_len[NRESTS / 4]; // atomic adds on 1-byte lengths
	__shared__ uint32_t lastword1[NSLOTS];
	__shared__ uint4 lastword2[NSLOTS];

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < NRESTS / 4)
		ht_len[threadid] = 0;

	uint32_t hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	uint32_t si[2];
	uint4 tt[2];
	uint32_t ta[2];

	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	__syncthreads();

	uint32_t bsize = umin(eq->edata.nslots[2][bucketid], NSLOTS);

	#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		SlotSmall &xs = eq->round2trees[bucketid].treessmall[si[i]];
		SlotTiny &xst = eq->round2trees[bucketid].treestiny[si[i]];

		tt[i] = *(uint4*)(&xs.hash[0]);
		lastword2[si[i]] = tt[i];
		ta[i] = xst.hash[0];
		lastword1[si[i]] = ta[i];
		asm("bfe.u32 %0, %1, 12, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		int shift = (hr[i] % 4) * 8;
		pos[i] = (atomicAdd(&ht_len[hr[i] / 4], 1U << shift) >> shift) & 0xff; // atomic adds on 1-byte lengths
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

			xors[4] = ta[i] ^ lastword1[p];

			if (xors[4] != 0)
			{
				*(uint4*)(&xors[0]) = tt[i] ^ lastword2[p];

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
				uint16_t prev = ht[hr[i]][k];

				int pair = __byte_perm(si[i], prev, 0x1054);
				uint32_t i = __byte_perm(pair, 0, 0x4510);
				uint32_t k = __byte_perm(pair, 0, 0x4532);

				xors[4] = lastword1[i] ^ lastword1[k];

				if (xors[4] != 0)
				{
					*(uint4*)(&xors[0]) = lastword2[i] ^ lastword2[k];

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
	}
}

} // namespace digit_3

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__forceinline__ void Digit_3(Equi<RB, SM>* equi)
{
	using namespace digit_3;

	kernel<RB, SM, SSM, PACKER, THREADS> << <NBUCKETS, THREADS >> >(equi);
}

