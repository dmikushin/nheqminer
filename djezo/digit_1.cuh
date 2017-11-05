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
	__shared__ uint ht_len[256];
	__shared__ uint2 lastword1[RB8_NSLOTS];
	__shared__ uint4 lastword2[RB8_NSLOTS];

	const int& threadid = threadIdx.x;
	const int& bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < 256)
		ht_len[threadid] = 0;
	
	__syncthreads();

	// Cache lastword1 and bit extraction.
	const Slot* s = eq->round0trees[bucketid];
	#pragma unroll 1
	for (uint32_t si = threadid, bsize = umin(eq->edata.nslots0[bucketid], RB8_NSLOTS); si < bsize; si += THREADS)
	{
		uint2 ta = *(uint2*)(&s[si].hash[4]);
		lastword1[si] = ta;

		uint8_t hr = ta.x >> 20;
		int pos = atomicAdd(&ht_len[hr], 1U);
		if (pos < (SSM - 1))
		{
			ht[hr][pos] = si;
			lastword2[si] = *(uint4*)(&s[si].hash[0]);
		}
	}

	__syncthreads();
	
	#pragma unroll 1
	for (int i = threadid, e = 256 * (SSM - 1); i < e; i += THREADS)
	{	
		int pos = i >> 8; // i / 256;
		int hr = i & 0xff; // i % 256;
		int len = min(ht_len[hr], SSM - 1);

		if (len < 2) continue;
		
		if (pos >= len) continue;

		#pragma unroll 1
		for (int si = ht[hr][pos], k = 0; k < pos; k++)
		{
			int ii = ht[hr][k];
			
			uint32_t xors[6];

			xors[0] = lastword1[ii].x ^ lastword1[si].x;

			uint32_t xorbucketid;
			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(RB), "r"(BUCKBITS));
			int xorslot = atomicAdd(&eq->edata.nslots[1][xorbucketid], 1);

			if (xorslot < NSLOTS)
			{
				xors[1] = lastword1[ii].y ^ lastword1[si].y;
				*(uint4*)(&xors[2]) = lastword2[ii] ^ lastword2[si];

				Slot &xs = eq->trees[0][xorbucketid][xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
				uint3 ttx;
				ttx.x = xors[5];
				ttx.y = xors[0];
				ttx.z = DefaultPacker::set_bucketid_and_slots(bucketid, ii, si, 8, RB8_NSLOTS);
				*(uint3*)(&xs.hash[4]) = ttx;
			}
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

