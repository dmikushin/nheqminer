/*
  Last round function is similar to previous ones but has different ending.
  We use warps to process final candidates. Each warp process one candidate.
  First two bidandsids (uint32_t of stored bucketid and two slotids) are retreived by
  lane 0 and lane 16, next four bidandsids by lane 0, 8, 16 and 24, ... until
  all lanes in warp have bidandsids from round 4. Next, each thread retreives
  16 indices. While doing so, indices are put into comparison using atomicExch
  to determine if there are duplicates (tromp's method). At the end, if no
  duplicates are found, candidate solution is saved (all indices). Note that this
  dup check method is not exact so CPU dup checking is needed after.
*/
template <uint32_t RB, uint32_t SM, int SSM, uint32_t FCT, typename PACKER, uint32_t MAXPAIRS, uint32_t DUPBITS, uint32_t W>
__global__ void digit_last_wdc(Equi<RB, SM>* eq)
{
	__shared__ uint8_t shared_data[8192];
	int* ht_len = (int*)(&shared_data[0]);
	int* pairs = ht_len;
	uint32_t* lastword = (uint32_t*)(&shared_data[256 * 4]);
	uint16_t* ht = (uint16_t*)(&shared_data[256 * 4 + RB8_NSLOTS_LD * 4]);
	uint32_t* pairs_len = (uint32_t*)(&shared_data[8188]);

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	// reset hashtable len
#pragma unroll
	for (uint32_t i = 0; i != FCT; ++i)
		ht_len[(i * (256 / FCT)) + threadid] = 0;

	if (threadid == ((256 / FCT) - 1))
		*pairs_len = 0;

	SlotTiny* buck = eq->treestiny[0][bucketid];
	uint32_t bsize = umin(eq->edata.nslots8[bucketid], RB8_NSLOTS_LD);

	uint32_t si[3 * FCT];
	uint32_t hr[3 * FCT];
	int pos[3 * FCT];
	uint32_t lw[3 * FCT];
#pragma unroll
	for (uint32_t i = 0; i != (3 * FCT); ++i)
		pos[i] = SSM;

	__syncthreads();

#pragma unroll
	for (uint32_t i = 0; i != (3 * FCT); ++i)
	{
		si[i] = i * (256 / FCT) + threadid;
		if (si[i] >= bsize) break;

		const SlotTiny* pslot1 = buck + si[i];

		// get xhash
		uint2 tt = *(uint2*)(&pslot1->hash[0]);
		lw[i] = tt.x;
		lastword[si[i]] = lw[i];

		uint32_t a;
		asm("bfe.u32 %0, %1, 20, 8;" : "=r"(a) : "r"(lw[i]));
		hr[i] = a;

		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1))
			ht[hr[i] * (SSM - 1) + pos[i]] = si[i];
	}

	__syncthreads();

#pragma unroll
	for (uint32_t i = 0; i != (3 * FCT); ++i)
	{
		if (pos[i] >= SSM) continue;

		for (int k = 0; k != pos[i]; ++k)
		{
			uint16_t prev = ht[hr[i] * (SSM - 1) + k];
			if (lw[i] != lastword[prev]) continue;
			uint32_t pindex = atomicAdd(pairs_len, 1);
			if (pindex >= MAXPAIRS) break;
			pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
		}
	}

	__syncthreads();
	uint32_t plen = umin(*pairs_len, 64);

#define CALC_LEVEL(a, b, c, d) { \
	uint32_t plvl = levels[b]; \
	uint32_t* bucks = eq->round4bidandsids[PACKER::get_bucketid(plvl, RB, SM)]; \
	uint32_t slot1 = PACKER::get_slot1(plvl, RB, SM); \
	uint32_t slot0 = PACKER::get_slot0(plvl, slot1, RB, SM); \
	levels[b] = bucks[slot1]; \
	levels[c] = bucks[slot0]; \
				}

#define CALC_LEVEL_SMALL(a, b, c, d) { \
	uint32_t plvl = levels[b]; \
	SlotSmall* bucks = eq->treessmall[a][PACKER::get_bucketid(plvl, RB, SM)]; \
	uint32_t slot1 = PACKER::get_slot1(plvl, RB, SM); \
	uint32_t slot0 = PACKER::get_slot0(plvl, slot1, RB, SM); \
	levels[b] = bucks[slot1].hash[d]; \
	levels[c] = bucks[slot0].hash[d]; \
				}

	uint32_t lane = threadIdx.x & 0x1f;
	uint32_t par = threadIdx.x >> 5;

	uint32_t* levels = (uint32_t*)&pairs[MAXPAIRS + (par << DUPBITS)];
	uint32_t* susp = levels;

	while (par < plen)
	{
		int pair = pairs[par];
		par += W;

		if (lane % 16 == 0)
		{
			uint32_t plvl;
			if (lane == 0) plvl = buck[__byte_perm(pair, 0, 0x4510)].hash[1];
			else plvl = buck[__byte_perm(pair, 0, 0x4532)].hash[1];
			SlotSmall* bucks = eq->treessmall[1][PACKER::get_bucketid(plvl, RB, SM)];
			uint32_t slot1 = PACKER::get_slot1(plvl, RB, SM);
			uint32_t slot0 = PACKER::get_slot0(plvl, slot1, RB, SM);
			levels[lane] = bucks[slot1].hash[2];
			levels[lane + 8] = bucks[slot0].hash[2];
		}

		if (lane % 8 == 0)
			CALC_LEVEL_SMALL(0, lane, lane + 4, 3);

		if (lane % 4 == 0)
			CALC_LEVEL_SMALL(2, lane, lane + 2, 3);

		if (lane % 2 == 0)
			CALC_LEVEL(0, lane, lane + 1, 4);

		uint32_t ind[16];

		uint32_t f1 = levels[lane];
		const SlotTiny* buck_v4 = &eq->round3trees[PACKER::get_bucketid(f1, RB, SM)].treestiny[0];
		const uint32_t slot1_v4 = PACKER::get_slot1(f1, RB, SM);
		const uint32_t slot0_v4 = PACKER::get_slot0(f1, slot1_v4, RB, SM);

		susp[lane] = 0xffffffff;
		susp[32 + lane] = 0xffffffff;

#define CHECK_DUP(a) \
	__any(atomicExch(&susp[(ind[a] & ((1 << DUPBITS) - 1))], (ind[a] >> DUPBITS)) == (ind[a] >> DUPBITS))

		uint32_t f2 = buck_v4[slot1_v4].hash[1];
		const SlotTiny* buck_v3_1 = &eq->round2trees[PACKER::get_bucketid(f2, RB, SM)].treestiny[0];
		const uint32_t slot1_v3_1 = PACKER::get_slot1(f2, RB, SM);
		const uint32_t slot0_v3_1 = PACKER::get_slot0(f2, slot1_v3_1, RB, SM);

		susp[64 + lane] = 0xffffffff;
		susp[96 + lane] = 0xffffffff;

		uint32_t f0 = buck_v3_1[slot1_v3_1].hash[1];
		const Slot* buck_v2_1 = eq->trees[0][PACKER::get_bucketid(f0, RB, SM)];
		const uint32_t slot1_v2_1 = PACKER::get_slot1(f0, RB, SM);
		const uint32_t slot0_v2_1 = PACKER::get_slot0(f0, slot1_v2_1, RB, SM);

		susp[128 + lane] = 0xffffffff;
		susp[160 + lane] = 0xffffffff;

		uint32_t f3 = buck_v2_1[slot1_v2_1].hash[6];
		const Slot* buck_fin_1 = eq->round0trees[DefaultPacker::get_bucketid(f3, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_1 = DefaultPacker::get_slot1(f3, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_1 = DefaultPacker::get_slot0(f3, slot1_fin_1, 8, RB8_NSLOTS);

		susp[192 + lane] = 0xffffffff;
		susp[224 + lane] = 0xffffffff;

		ind[0] = buck_fin_1[slot1_fin_1].hash[7];
		if (CHECK_DUP(0)) continue;
		ind[1] = buck_fin_1[slot0_fin_1].hash[7];
		if (CHECK_DUP(1)) continue;

		uint32_t f4 = buck_v2_1[slot0_v2_1].hash[6];
		const Slot* buck_fin_2 = eq->round0trees[DefaultPacker::get_bucketid(f4, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_2 = DefaultPacker::get_slot1(f4, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_2 = DefaultPacker::get_slot0(f4, slot1_fin_2, 8, RB8_NSLOTS);

		ind[2] = buck_fin_2[slot1_fin_2].hash[7];
		if (CHECK_DUP(2)) continue;
		ind[3] = buck_fin_2[slot0_fin_2].hash[7];
		if (CHECK_DUP(3)) continue;

		uint32_t f5 = buck_v3_1[slot0_v3_1].hash[1];
		const Slot* buck_v2_2 = eq->trees[0][PACKER::get_bucketid(f5, RB, SM)];
		const uint32_t slot1_v2_2 = PACKER::get_slot1(f5, RB, SM);
		const uint32_t slot0_v2_2 = PACKER::get_slot0(f5, slot1_v2_2, RB, SM);

		uint32_t f6 = buck_v2_2[slot1_v2_2].hash[6];
		const Slot* buck_fin_3 = eq->round0trees[DefaultPacker::get_bucketid(f6, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_3 = DefaultPacker::get_slot1(f6, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_3 = DefaultPacker::get_slot0(f6, slot1_fin_3, 8, RB8_NSLOTS);

		ind[4] = buck_fin_3[slot1_fin_3].hash[7];
		if (CHECK_DUP(4)) continue;
		ind[5] = buck_fin_3[slot0_fin_3].hash[7];
		if (CHECK_DUP(5)) continue;

		uint32_t f7 = buck_v2_2[slot0_v2_2].hash[6];
		const Slot* buck_fin_4 = eq->round0trees[DefaultPacker::get_bucketid(f7, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_4 = DefaultPacker::get_slot1(f7, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_4 = DefaultPacker::get_slot0(f7, slot1_fin_4, 8, RB8_NSLOTS);

		ind[6] = buck_fin_4[slot1_fin_4].hash[7];
		if (CHECK_DUP(6)) continue;
		ind[7] = buck_fin_4[slot0_fin_4].hash[7];
		if (CHECK_DUP(7)) continue;

		uint32_t f8 = buck_v4[slot0_v4].hash[1];
		const SlotTiny* buck_v3_2 = &eq->round2trees[PACKER::get_bucketid(f8, RB, SM)].treestiny[0];
		const uint32_t slot1_v3_2 = PACKER::get_slot1(f8, RB, SM);
		const uint32_t slot0_v3_2 = PACKER::get_slot0(f8, slot1_v3_2, RB, SM);

		uint32_t f9 = buck_v3_2[slot1_v3_2].hash[1];
		const Slot* buck_v2_3 = eq->trees[0][PACKER::get_bucketid(f9, RB, SM)];
		const uint32_t slot1_v2_3 = PACKER::get_slot1(f9, RB, SM);
		const uint32_t slot0_v2_3 = PACKER::get_slot0(f9, slot1_v2_3, RB, SM);

		uint32_t f10 = buck_v2_3[slot1_v2_3].hash[6];
		const Slot* buck_fin_5 = eq->round0trees[DefaultPacker::get_bucketid(f10, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_5 = DefaultPacker::get_slot1(f10, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_5 = DefaultPacker::get_slot0(f10, slot1_fin_5, 8, RB8_NSLOTS);

		ind[8] = buck_fin_5[slot1_fin_5].hash[7];
		if (CHECK_DUP(8)) continue;
		ind[9] = buck_fin_5[slot0_fin_5].hash[7];
		if (CHECK_DUP(9)) continue;

		uint32_t f11 = buck_v2_3[slot0_v2_3].hash[6];
		const Slot* buck_fin_6 = eq->round0trees[DefaultPacker::get_bucketid(f11, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_6 = DefaultPacker::get_slot1(f11, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_6 = DefaultPacker::get_slot0(f11, slot1_fin_6, 8, RB8_NSLOTS);

		ind[10] = buck_fin_6[slot1_fin_6].hash[7];
		if (CHECK_DUP(10)) continue;
		ind[11] = buck_fin_6[slot0_fin_6].hash[7];
		if (CHECK_DUP(11)) continue;

		uint32_t f12 = buck_v3_2[slot0_v3_2].hash[1];
		const Slot* buck_v2_4 = eq->trees[0][PACKER::get_bucketid(f12, RB, SM)];
		const uint32_t slot1_v2_4 = PACKER::get_slot1(f12, RB, SM);
		const uint32_t slot0_v2_4 = PACKER::get_slot0(f12, slot1_v2_4, RB, SM);

		uint32_t f13 = buck_v2_4[slot1_v2_4].hash[6];
		const Slot* buck_fin_7 = eq->round0trees[DefaultPacker::get_bucketid(f13, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_7 = DefaultPacker::get_slot1(f13, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_7 = DefaultPacker::get_slot0(f13, slot1_fin_7, 8, RB8_NSLOTS);

		ind[12] = buck_fin_7[slot1_fin_7].hash[7];
		if (CHECK_DUP(12)) continue;
		ind[13] = buck_fin_7[slot0_fin_7].hash[7];
		if (CHECK_DUP(13)) continue;

		uint32_t f14 = buck_v2_4[slot0_v2_4].hash[6];
		const Slot* buck_fin_8 = eq->round0trees[DefaultPacker::get_bucketid(f14, 8, RB8_NSLOTS)];
		const uint32_t slot1_fin_8 = DefaultPacker::get_slot1(f14, 8, RB8_NSLOTS);
		const uint32_t slot0_fin_8 = DefaultPacker::get_slot0(f14, slot1_fin_8, 8, RB8_NSLOTS);

		ind[14] = buck_fin_8[slot1_fin_8].hash[7];
		if (CHECK_DUP(14)) continue;
		ind[15] = buck_fin_8[slot0_fin_8].hash[7];
		if (CHECK_DUP(15)) continue;

		uint32_t soli;
		if (lane == 0)
		{
			soli = atomicAdd(&eq->edata.solutions.nsols, 1);
		}
		soli = __shfl(soli, 0);

		if (soli < MAXREALSOLS)
		{
			uint32_t pos = lane << 4;
			*(uint4*)(&eq->edata.solutions.sols[soli][pos]) = *(uint4*)(&ind[0]);
			*(uint4*)(&eq->edata.solutions.sols[soli][pos + 4]) = *(uint4*)(&ind[4]);
			*(uint4*)(&eq->edata.solutions.sols[soli][pos + 8]) = *(uint4*)(&ind[8]);
			*(uint4*)(&eq->edata.solutions.sols[soli][pos + 12]) = *(uint4*)(&ind[12]);
		}
	}
}

