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
__global__ void kernel(Equi<RB, SM>* eq)
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
	
	uint16_t* htp = (uint16_t*)ht;
	for (int i = threadid, e = 256 * (SSM - 1); i < e; i += THREADS)
		htp[i] = USHRT_MAX;

	__syncthreads();

	for (uint32_t si = threadid, bsize = umin(eq->edata.nslots0[bucketid], RB8_NSLOTS); si < bsize; si += THREADS)
	{
		const Slot* pslot1 = &eq->round0trees[bucketid][si];

		// get xhash
		uint4 tb = *(uint4*)(&pslot1->hash[0]);
		uint2 ta = *(uint2*)(&pslot1->hash[4]);
		lastword1[si] = ta;
		lastword2[si] = tb;

		uint32_t hr;
		asm("bfe.u32 %0, %1, 20, 8;" : "=r"(hr) : "r"(ta.x));
		int shift = (hr % 4) * 8;
		int pos = (atomicAdd(&ht_len[hr / 4], 1U << shift) >> shift) & 0xff; // atomic adds on 1-byte lengths
		if (pos < (SSM - 1)) ht[hr][pos] = si;
	}

	__syncthreads();

	for (int i = threadid, e = 256 * (SSM - 1); i < e; i += THREADS)
	{
		int pos = i / 256;
		int hr = i - pos * 256; // i % 256;

		int si = ht[hr][pos];
		if (si == USHRT_MAX) continue;

		for (int k = 0, pair = 0, ii = si, kk = ht[hr][0]; k < pos;
			pair = __byte_perm(si, ht[hr][k], 0x1054),
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
			
			if (k == pos) break;
		}
	}
}

} // namespace digit_1

template <uint32_t RB, uint32_t SM, int SSM, uint32_t THREADS>
__forceinline__ void Digit_1(Equi<RB, SM>* equi)
{
	using namespace digit_1;

	kernel<RB, SM, SSM, 512> << <4096, 512 >> >(equi);
}

