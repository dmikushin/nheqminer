template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS>
__global__ void digit_8(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][(SSM - 1)];
	__shared__ uint32_t lastword[NSLOTS][2];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t bsize_sh;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	ht_len[threadid] = 0;
	if (threadid == (NRESTS - 1))
	{
		next_pair = 0;
		pairs_len = 0;
	}
	else if (threadid == (NRESTS - 33))
		bsize_sh = umin(eq->edata.nslots[7][bucketid], NSLOTS);

	SlotSmall* buck = eq->treessmall[1][bucketid];

	uint32_t hr[3];
	int pos[3];
	pos[0] = pos[1] = pos[2] = SSM;

	uint32_t si[3];
	uint2 tt[3];

	__syncthreads();

	uint32_t bsize = bsize_sh;

#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		si[i] = i * NRESTS + threadid;
		if (si[i] >= bsize) break;

		const SlotSmall* pslot1 = buck + si[i];

		// get xhash
		tt[i] = *(uint2*)(&pslot1->hash[0]);
		*(uint2*)(&lastword[si[i]][0]) = *(uint2*)(&tt[i].x);
		asm("bfe.u32 %0, %1, 8, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	uint32_t xors[2];
	uint32_t bexor, xorbucketid, xorslot;

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
				bexor = __byte_perm(xors[0], xors[1], 0x0765);
				xorbucketid = bexor >> (12 + 8);
				xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
				if (xorslot < RB8_NSLOTS_LD)
				{
					SlotTiny &xs = eq->treestiny[0][xorbucketid][xorslot];
					uint2 tt;
					tt.x = xors[1];
					tt.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint2*)(&xs.hash[0]) = tt;
				}
			}

			if (pos[i] > 1)
			{
				p = ht[hr[i]][1];

				*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

				if (xors[1] != 0)
				{
					bexor = __byte_perm(xors[0], xors[1], 0x0765);
					xorbucketid = bexor >> (12 + 8);
					xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
					if (xorslot < RB8_NSLOTS_LD)
					{
						SlotTiny &xs = eq->treestiny[0][xorbucketid][xorslot];
						uint2 tt;
						tt.x = xors[1];
						tt.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
						*(uint2*)(&xs.hash[0]) = tt;
					}
				}

				for (int k = 2; k != pos[i]; ++k)
				{
					uint32_t pindex = atomicAdd(&pairs_len, 1);
					if (pindex >= MAXPAIRS) break;
					uint16_t prev = ht[hr[i]][k];
					pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
				}
			}
		}
	}

	__syncthreads();

	// process pairs
	uint32_t plen = umin(pairs_len, MAXPAIRS);
	for (uint32_t s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		uint32_t i = __byte_perm(pair, 0, 0x4510);
		uint32_t k = __byte_perm(pair, 0, 0x4532);

		*(uint2*)(&xors[0]) = *(uint2*)(&lastword[i][0]) ^ *(uint2*)(&lastword[k][0]);

		if (xors[1] == 0)
			continue;

		bexor = __byte_perm(xors[0], xors[1], 0x0765);
		xorbucketid = bexor >> (12 + 8);
		xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
		if (xorslot >= RB8_NSLOTS_LD) continue;
		SlotTiny &xs = eq->treestiny[0][xorbucketid][xorslot];
		uint2 tt;
		tt.x = xors[1];
		tt.y = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
		*(uint2*)(&xs.hash[0]) = tt;
	}
}

