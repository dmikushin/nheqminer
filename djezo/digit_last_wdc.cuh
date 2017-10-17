#include <cuda.h> // CUDA_VERSION

namespace digit_last_wdc {

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
__global__ void kernel(Equi<RB, SM>* eq)
{
	__shared__ uint8_t shared_data[8192];
	int* ht_len = (int*)(&shared_data[0]);
	int* pairs = ht_len;
	uint32_t* lastword = (uint32_t*)(&shared_data[256]);
	uint16_t* ht = (uint16_t*)(&shared_data[256 + RB8_NSLOTS_LD * 4]);
	uint32_t* pairs_len = (uint32_t*)(&shared_data[8188]);

	const uint32_t threadid = threadIdx.x;
	const uint32_t bucketid = blockIdx.x;

	for (uint32_t i = threadid; i < 256; i += blockDim.x)
		ht_len[i] = 0;

	if (threadid == ((256 / FCT) - 1))
		*pairs_len = 0;

	SlotTiny* buck = eq->treestiny[0][bucketid];

	uint32_t hr[3 * FCT];
	int pos[3 * FCT];

	#pragma unroll
	for (uint32_t i = 0; i != (3 * FCT); ++i)
		pos[i] = SSM;

	__syncthreads();

	uint32_t bsize = umin(eq->edata.nslots8[bucketid], RB8_NSLOTS_LD);

	#pragma unroll
	for (uint32_t i = 0; i != (3 * FCT); ++i)
	{
		int si = i * (256 / FCT) + threadid;
		if (si >= bsize) break;

		const SlotTiny* pslot1 = buck + si;

		// get xhash
		uint tt = *(uint*)(&pslot1->hash[0]);
		lastword[si] = tt;
		asm("bfe.u32 %0, %1, 20, 8;" : "=r"(hr[i]) : "r"(tt));
		int shift = (hr[i] % 4) * 8;
		pos[i] = (atomicAdd(&ht_len[hr[i] / 4], 1U << shift) >> shift) & 0xff; // atomic adds on 1-byte lengths
		if (pos[i] < (SSM - 1)) ht[hr[i] * (SSM - 1) + pos[i]] = si;
	}

	__syncthreads();

	#pragma unroll
	for (uint32_t i = 0; i != (3 * FCT); ++i)
	{
		if (pos[i] >= SSM) continue;

		int si = i * (256 / FCT) + threadid;
		const uint32_t& lw = lastword[si];

		for (int k = 0; k != pos[i]; ++k)
		{
			uint16_t prev = ht[hr[i] * (SSM - 1) + k];
			if (lw != lastword[prev]) continue;
			uint32_t pindex = atomicAdd(pairs_len, 1);
			if (pindex >= MAXPAIRS) break;
			pairs[pindex] = __byte_perm(si, prev, 0x1054);
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

#if CUDA_VERSION >= 9000
#define CHECK_DUP(a) \
	__any_sync(0xffffffff, atomicExch(&susp[(ind[a] & ((1 << DUPBITS) - 1))], (ind[a] >> DUPBITS)) == (ind[a] >> DUPBITS))
#else
#define CHECK_DUP(a) \
	__any(atomicExch(&susp[(ind[a] & ((1 << DUPBITS) - 1))], (ind[a] >> DUPBITS)) == (ind[a] >> DUPBITS))
#endif

		uint32_t f1 = levels[lane];
		const SlotTiny* buck_v4 = &eq->round3trees[PACKER::get_bucketid(f1, RB, SM)].treestiny[0];
		const uint32_t slot1_v4 = PACKER::get_slot1(f1, RB, SM);
		const uint32_t slot0_v4 = PACKER::get_slot0(f1, slot1_v4, RB, SM);

		uint32_t f2 = buck_v4[slot1_v4].hash[1];
		const SlotTiny* buck_v3_1 = &eq->round2trees[PACKER::get_bucketid(f2, RB, SM)].treestiny[0];
		const uint32_t slot1_v3_1 = PACKER::get_slot1(f2, RB, SM);
		const uint32_t slot0_v3_1 = PACKER::get_slot0(f2, slot1_v3_1, RB, SM);

		uint32_t f0 = buck_v3_1[slot1_v3_1].hash[1];
		const Slot* buck_v2_1 = eq->trees[0][PACKER::get_bucketid(f0, RB, SM)];
		const uint32_t slot1_v2_1 = PACKER::get_slot1(f0, RB, SM);
		const uint32_t slot0_v2_1 = PACKER::get_slot0(f0, slot1_v2_1, RB, SM);

		uint32_t f5 = buck_v3_1[slot0_v3_1].hash[1];
		const Slot* buck_v2_2 = eq->trees[0][PACKER::get_bucketid(f5, RB, SM)];
		const uint32_t slot1_v2_2 = PACKER::get_slot1(f5, RB, SM);
		const uint32_t slot0_v2_2 = PACKER::get_slot0(f5, slot1_v2_2, RB, SM);

		uint32_t f8 = buck_v4[slot0_v4].hash[1];
		const SlotTiny* buck_v3_2 = &eq->round2trees[PACKER::get_bucketid(f8, RB, SM)].treestiny[0];
		const uint32_t slot1_v3_2 = PACKER::get_slot1(f8, RB, SM);
		const uint32_t slot0_v3_2 = PACKER::get_slot0(f8, slot1_v3_2, RB, SM);

		uint32_t f9 = buck_v3_2[slot1_v3_2].hash[1];
		const Slot* buck_v2_3 = eq->trees[0][PACKER::get_bucketid(f9, RB, SM)];
		const uint32_t slot1_v2_3 = PACKER::get_slot1(f9, RB, SM);
		const uint32_t slot0_v2_3 = PACKER::get_slot0(f9, slot1_v2_3, RB, SM);

		uint32_t f12 = buck_v3_2[slot0_v3_2].hash[1];
		const Slot* buck_v2_4 = eq->trees[0][PACKER::get_bucketid(f12, RB, SM)];
		const uint32_t slot1_v2_4 = PACKER::get_slot1(f12, RB, SM);
		const uint32_t slot0_v2_4 = PACKER::get_slot0(f12, slot1_v2_4, RB, SM);

		uint32_t ind[16] = { buck_v2_1[slot1_v2_1].hash[6], 0, buck_v2_1[slot0_v2_1].hash[6], 0, buck_v2_2[slot1_v2_2].hash[6], 0, buck_v2_2[slot0_v2_2].hash[6], 0, buck_v2_3[slot1_v2_3].hash[6], 0, buck_v2_3[slot0_v2_3].hash[6], 0, buck_v2_4[slot1_v2_4].hash[6], 0, buck_v2_4[slot0_v2_4].hash[6], 0 };

		susp[lane] = 0xffffffff;
		susp[32 + lane] = 0xffffffff;
		susp[64 + lane] = 0xffffffff;
		susp[96 + lane] = 0xffffffff;
		susp[128 + lane] = 0xffffffff;
		susp[160 + lane] = 0xffffffff;
		susp[192 + lane] = 0xffffffff;
		susp[224 + lane] = 0xffffffff;

		#pragma nounroll
		for (int i = 0; i < 7; i++)
		{
			const uint32_t& f = ind[2 * i];
			const Slot* buck_fin = eq->round0trees[DefaultPacker::get_bucketid(f, 8, RB8_NSLOTS)];
			uint32_t slot1_fin = DefaultPacker::get_slot1(f, 8, RB8_NSLOTS);
			uint32_t slot0_fin = DefaultPacker::get_slot0(f, slot1_fin, 8, RB8_NSLOTS);

			ind[2 * i] = buck_fin[slot1_fin].hash[7];
			if (CHECK_DUP(2 * i)) continue;
			ind[2 * i + 1] = buck_fin[slot0_fin].hash[7];
			if (CHECK_DUP(2 * i + 1)) continue;
		}

		{
			int i = 7;
			const uint32_t& f = ind[2 * i];
			const Slot* buck_fin = eq->round0trees[DefaultPacker::get_bucketid(f, 8, RB8_NSLOTS)];
			uint32_t slot1_fin = DefaultPacker::get_slot1(f, 8, RB8_NSLOTS);
			uint32_t slot0_fin = DefaultPacker::get_slot0(f, slot1_fin, 8, RB8_NSLOTS);

			ind[2 * i] = buck_fin[slot1_fin].hash[7];
			if (CHECK_DUP(2 * i)) continue;
			ind[2 * i + 1] = buck_fin[slot0_fin].hash[7];
			if (CHECK_DUP(2 * i + 1)) continue;
		}

		uint32_t soli;
		if (lane == 0)
		{
			soli = atomicAdd(&eq->edata.solutions.nsols, 1);
		}
#if CUDA_VERSION >= 9000
		soli = __shfl_sync(0xffffffff, soli, 0);
#else
		soli = __shfl(soli, 0);
#endif

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

} // namespace digit_last_wdc

template <uint32_t RB, uint32_t SM, int SSM, uint32_t FCT, typename PACKER, uint32_t MAXPAIRS, uint32_t DUPBITS, uint32_t W>
__forceinline__ void DigitLastWDC(Equi<RB, SM>* equi)
{
	using namespace digit_last_wdc;

	kernel<RB, SM, SSM, FCT, PACKER, MAXPAIRS, DUPBITS, W> << <4096, 256 / 2 >> >(equi);
}

