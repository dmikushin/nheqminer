namespace digit_7 {

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER>
__global__ void kernel(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][SSM - 1];
	__shared__ uint ht_len[NRESTS];
	__shared__ uint32_t lastword[NSLOTS][2];

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	if (threadid < NRESTS)
		ht_len[threadid] = 0;

	SlotSmall* buck = eq->treessmall[0][bucketid];

	uint32_t hr[3];
	int pos[3];
	pos[0] = pos[1] = pos[2] = SSM;

	uint32_t si[3];
	uint4 tt[3];

	__syncthreads();

	uint32_t bsize = umin(eq->edata.nslots[6][bucketid], NSLOTS);

	#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		si[i] = i * NRESTS + threadid;
		if (si[i] >= bsize) break;

		const SlotSmall* pslot1 = buck + si[i];

		// get xhash
		tt[i] = *(uint4*)(&pslot1->hash[0]);
		*(uint2*)(&lastword[si[i]][0]) = *(uint2*)(&tt[i].x);
		asm("bfe.u32 %0, %1, 12, %2;" : "=r"(hr[i]) : "r"(tt[i].z), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1U);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	uint32_t xors[2];
	uint32_t xorbucketid, xorslot;

	#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

			if (xors[1] != 0)
			{
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(8 + RB), "r"(BUCKBITS));
				xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					SlotSmall &xs = eq->treessmall[1][xorbucketid][xorslot];
					uint4 ttx;
					ttx.x = xors[0];
					ttx.y = xors[1];
					ttx.z = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					ttx.w = 0;
					*(uint4*)(&xs.hash[0]) = ttx;
				}
			}

			if (pos[i] > 1)
			{
				p = ht[hr[i]][1];

				*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

				if (xors[1] != 0)
				{
					asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(8 + RB), "r"(BUCKBITS));
					xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
					if (xorslot < NSLOTS)
					{
						SlotSmall &xs = eq->treessmall[1][xorbucketid][xorslot];
						uint4 ttx;
						ttx.x = xors[0];
						ttx.y = xors[1];
						ttx.z = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
						ttx.w = 0;
						*(uint4*)(&xs.hash[0]) = ttx;
					}
				}

				for (int k = 2; k != pos[i]; ++k)
				{
					uint16_t prev = ht[hr[i]][k];

					int pair = __byte_perm(si[i], prev, 0x1054);
					uint32_t i = __byte_perm(pair, 0, 0x4510);
					uint32_t k = __byte_perm(pair, 0, 0x4532);

					*(uint2*)(&xors[0]) = *(uint2*)(&lastword[i][0]) ^ *(uint2*)(&lastword[k][0]);

					if (xors[1] == 0)
						continue;

					asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(8 + RB), "r"(BUCKBITS));
					xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
					if (xorslot >= NSLOTS) continue;
					SlotSmall &xs = eq->treessmall[1][xorbucketid][xorslot];
					uint4 tt;
					tt.x = xors[0];
					tt.y = xors[1];
					tt.z = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
					tt.w = 0;
					*(uint4*)(&xs.hash[0]) = tt;
				}
			}
		}
	}
}

} // namespace digit_7

template <uint32_t RB, uint32_t SM, int SSM, typename PACKER>
__forceinline__ void Digit_7(Equi<RB, SM>* equi)
{
	using namespace digit_7;

	kernel<RB, SM, SSM, PACKER> << <NBUCKETS, NRESTS >> >(equi);
}

