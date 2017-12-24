namespace digit_2 {

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t THREADS>
__global__ void kernelDigit_2(Equi<RB, SM>* eq)
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

	for (uint32_t si = threadid, bsize = umin(eq->edata.nslots[1][bucketid], NSLOTS); si < bsize; si += THREADS)
	{
		const Slot* pslot1 = &eq->trees[0][bucketid][si];

		// get xhash
		uint4 ttx = *(uint4*)(&pslot1->hash[0]);
		lastword1[si] = ttx.x;
		uint2 tty = *(uint2*)(&pslot1->hash[4]);
		uint4 tt;
		tt.x = ttx.y;
		tt.y = ttx.z;
		tt.z = ttx.w;
		tt.w = tty.x;
		lastword2[si] = tt;

		uint32_t hr = tty.y & RESTMASK;
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

			xors[0] = lastword1[ii] ^ lastword1[kk];

			uint32_t xorbucketid = xors[0] >> (12 + RB);
			int xorslot = atomicAdd(&eq->edata.nslots[2][xorbucketid], 1);

			if (xorslot < NSLOTS)
			{
				*(uint4*)(&xors[1]) = lastword2[ii] ^ lastword2[kk];
				SlotSmall &xs = eq->round2trees[xorbucketid].treessmall[xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[0]);
				SlotTiny &xst = eq->round2trees[xorbucketid].treestiny[xorslot];
				uint2 ttx;
				ttx.x = xors[4];
				ttx.y = PACKER::set_bucketid_and_slots(bucketid, ii, kk, RB, SM);
				*(uint2*)(&xst.hash[0]) = ttx;
			}
			
			k++;
			
			if (k == pos) break;
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

