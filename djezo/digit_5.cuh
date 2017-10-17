namespace digit_5 {

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__global__ void kernel(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][SSM - 1];
	__shared__ uint ht_len[NRESTS / 4]; // atomic adds on 1-byte lengths
	__shared__ uint4 lastword[NSLOTS];

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	if (threadid < NRESTS / 4)
		ht_len[threadid] = 0;

	SlotSmall* buck = eq->treessmall[3][bucketid];
	uint32_t bsize = umin(eq->edata.nslots[4][bucketid], NSLOTS);

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

		const SlotSmall* pslot1 = buck + si[i];

		tt[i] = *(uint4*)(&pslot1->hash[0]);
		lastword[si[i]] = tt[i];
		asm("bfe.u32 %0, %1, 4, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		int shift = (hr[i] % 4) * 8;
		pos[i] = (atomicAdd(&ht_len[hr[i] / 4], 1U << shift) >> shift) & 0xff; // atomic adds on 1-byte lengths
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
				uint16_t prev = ht[hr[i]][k];

				int pair = __byte_perm(si[i], prev, 0x1054);
				uint32_t i = __byte_perm(pair, 0, 0x4510);
				uint32_t k = __byte_perm(pair, 0, 0x4532);

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
	}
}

} // namespace digit_4

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__forceinline__ void Digit_5(Equi<RB, SM>* equi)
{
	using namespace digit_5;

	kernel<RB, SM, SSM, PACKER, THREADS> << <NBUCKETS, THREADS >> >(equi);
}

