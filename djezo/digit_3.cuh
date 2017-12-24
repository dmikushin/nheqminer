namespace digit_3 {

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__global__ void kernel(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][SSM - 1];
	__shared__ uint ht_len[NRESTS];
	__shared__ uint32_t lastword1[NSLOTS];
	__shared__ uint4 lastword2[NSLOTS];

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < NRESTS)
		ht_len[threadid] = 0;

	uint16_t* htp = (uint16_t*)ht;
	for (int i = threadid, e = NRESTS * (SSM - 1); i < e; i += THREADS)
		htp[i] = USHRT_MAX;

	__syncthreads();

	for (uint32_t si = threadid, bsize = umin(eq->edata.nslots[2][bucketid], NSLOTS); si < bsize; si += THREADS)
	{
		SlotSmall &xs = eq->round2trees[bucketid].treessmall[si];
		SlotTiny &xst = eq->round2trees[bucketid].treestiny[si];

		// get xhash
		uint4 tt = *(uint4*)(&xs.hash[0]);
		lastword2[si] = tt;
		lastword1[si] = xst.hash[0];
		
		uint32_t hr;
		asm("bfe.u32 %0, %1, 12, %2;" : "=r"(hr) : "r"(tt.x), "r"(RB));
		int pos = atomicAdd(&ht_len[hr], 1U);
		if (pos < (SSM - 1)) ht[hr][pos] = si;
	}

	__syncthreads();

	for (int i = threadid, e = NRESTS * (SSM - 1); i < e; i += THREADS)
	{
		int pos = i / NRESTS;
		int hr = i - pos * NRESTS; // i % NRESTS;

		int si = ht[hr][pos];
		if (si == USHRT_MAX) continue;

		for (int k = 0, pair = 0, ii = si, kk = ht[hr][0]; k < pos;
			pair = __byte_perm(si, ht[hr][k], 0x1054),
			ii = __byte_perm(pair, 0, 0x4510),
			kk = __byte_perm(pair, 0, 0x4532))
		{
			uint32_t xors[5];

			xors[4] = lastword1[ii] ^ lastword1[kk];

			if (xors[4] != 0)
			{
				*(uint4*)(&xors[0]) = lastword2[ii] ^ lastword2[kk];

				uint32_t xorbucketid;
				uint32_t bexor = __byte_perm(xors[0], xors[1], 0x2107);
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(bexor), "r"(RB), "r"(BUCKBITS));
				int xorslot = atomicAdd(&eq->edata.nslots[3][xorbucketid], 1);

				if (xorslot < NSLOTS)
				{
					SlotSmall &xs = eq->round3trees[xorbucketid].treessmall[xorslot];
					*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
					SlotTiny &xst = eq->round3trees[xorbucketid].treestiny[xorslot];
					uint2 ttx;
					ttx.x = bexor;
					ttx.y = PACKER::set_bucketid_and_slots(bucketid, ii, kk, RB, SM);
					*(uint2*)(&xst.hash[0]) = ttx;
				}
			}

			k++;
			
			if (k == pos) break;
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

