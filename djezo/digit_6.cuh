template <uint32_t RB, uint32_t SM, int SSM, typename PACKER, uint32_t MAXPAIRS>
__global__ void digit_6(Equi<RB, SM>* eq)
{
	__shared__ uint16_t ht[NRESTS][(SSM - 1)];
	__shared__ uint2 lastword1[NSLOTS];
	__shared__ uint32_t lastword2[NSLOTS];
	__shared__ int ht_len[MAXPAIRS];
	__shared__ uint32_t pairs_len;
	__shared__ uint32_t bsize_sh;
	__shared__ uint32_t next_pair;

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
	ht_len[threadid] = 0;
	if (threadid == (NRESTS - 1))
	{
		pairs_len = 0;
		next_pair = 0;
	}
	else if (threadid == (NRESTS - 33))
		bsize_sh = umin(eq->edata.nslots[5][bucketid], NSLOTS);

	SlotSmall* buck = eq->treessmall[2][bucketid];

	uint32_t hr[3];
	int pos[3];
	pos[0] = pos[1] = pos[2] = SSM;

	uint32_t si[3];
	uint4 tt[3];

	__syncthreads();

	uint32_t bsize = bsize_sh;

#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		si[i] = i * NRESTS + threadid;
		if (si[i] >= bsize) break;

		const SlotSmall* pslot1 = buck + si[i];

		tt[i] = *(uint4*)(&pslot1->hash[0]);
		lastword1[si[i]] = *(uint2*)(&tt[i].x);
		lastword2[si[i]] = tt[i].z;
		asm("bfe.u32 %0, %1, 16, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	// doing this to save shared memory
	int* pairs = ht_len;
	__syncthreads();

	uint32_t xors[3];
	uint32_t bexor, xorbucketid, xorslot;

#pragma unroll
	for (uint32_t i = 0; i != 3; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			uint16_t p = ht[hr[i]][0];

			xors[2] = tt[i].z ^ lastword2[p];

			if (xors[2] != 0)
			{
				*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ lastword1[p];

				bexor = __byte_perm(xors[0], xors[1], 0x1076);
				xorbucketid = bexor >> (12 + RB);
				xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					SlotSmall &xs = eq->treessmall[0][xorbucketid][xorslot];
					uint4 ttx;
					ttx.x = xors[1];
					ttx.y = xors[2];
					ttx.z = bexor;
					ttx.w = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint4*)(&xs.hash[0]) = ttx;
				}
			}

			if (pos[i] > 1)
			{
				p = ht[hr[i]][1];

				xors[2] = tt[i].z ^ lastword2[p];

				if (xors[2] != 0)
				{
					*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ lastword1[p];

					bexor = __byte_perm(xors[0], xors[1], 0x1076);
					xorbucketid = bexor >> (12 + RB);
					xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
					if (xorslot < NSLOTS)
					{
						SlotSmall &xs = eq->treessmall[0][xorbucketid][xorslot];
						uint4 ttx;
						ttx.x = xors[1];
						ttx.y = xors[2];
						ttx.z = bexor;
						ttx.w = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
						*(uint4*)(&xs.hash[0]) = ttx;
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
		uint32_t pair = pairs[s];
		uint32_t i = __byte_perm(pair, 0, 0x4510);
		uint32_t k = __byte_perm(pair, 0, 0x4532);

		xors[2] = lastword2[i] ^ lastword2[k];
		if (xors[2] == 0)
			continue;

		*(uint2*)(&xors[0]) = lastword1[i] ^ lastword1[k];

		bexor = __byte_perm(xors[0], xors[1], 0x1076);
		xorbucketid = bexor >> (12 + RB);
		xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
		if (xorslot >= NSLOTS) continue;
		SlotSmall &xs = eq->treessmall[0][xorbucketid][xorslot];
		uint4 ttx;
		ttx.x = xors[1];
		ttx.y = xors[2];
		ttx.z = bexor;
		ttx.w = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
		*(uint4*)(&xs.hash[0]) = ttx;
	}
}

