namespace digit_2 {

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__global__ void kernelDigit_2(Equi<RB, SM>* eq)
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

	uint32_t ta[2];
	uint4 tt[2];

	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	__syncthreads();

	uint32_t bsize = umin(eq->edata.nslots[1][bucketid], NSLOTS);

	#pragma unroll
	for (uint32_t i = 0; i != 2; ++i)
	{
		uint32_t si = i * THREADS + threadid;
		if (si >= bsize) break;

		const Slot* pslot1 = &eq->trees[0][bucketid][si];

		uint4 ttx = *(uint4*)(&pslot1->hash[0]);
		lastword1[si] = ta[i] = ttx.x;
		uint2 tty = *(uint2*)(&pslot1->hash[4]);
		tt[i].x = ttx.y;
		tt[i].y = ttx.z;
		tt[i].z = ttx.w;
		tt[i].w = tty.x;
		lastword2[si] = tt[i];

		hr[i] = tty.y & RESTMASK;
		int shift = (hr[i] % 4) * 8;
		pos[i] = (atomicAdd(&ht_len[hr[i] / 4], 1U << shift) >> shift) & 0xff; // atomic adds on 1-byte lengths
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si;
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
			uint32_t si = i * THREADS + threadid;

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
				ttx.y = PACKER::set_bucketid_and_slots(bucketid, si, p, RB, SM);
				*(uint2*)(&xst.hash[0]) = ttx;
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				uint16_t prev = ht[hr[i]][k];
				int pair = __byte_perm(si, prev, 0x1054);

				uint32_t i = __byte_perm(pair, 0, 0x4510);
				uint32_t k = __byte_perm(pair, 0, 0x4532);

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
	}
}

} // namespace digit_2

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__forceinline__ void Digit_2(Equi<RB, SM>* equi)
{
	using namespace digit_2;

	kernelDigit_2<RB, SM, SSM, PACKER, THREADS> << <NBUCKETS, THREADS >> >(equi);
}

